import { Router, type IRouter, type Request, type Response } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { logger } from "../lib/logger";

const router: IRouter = Router();

const openaiClient = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY ?? "dummy",
});

const anthropicClient = new Anthropic({
  baseURL: process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY ?? "dummy",
});

const OPENAI_MODELS = ["gpt-5.2", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3"];
const ANTHROPIC_MODELS = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"];

function verifyToken(req: Request, res: Response): boolean {
  const proxyKey = process.env.PROXY_API_KEY;
  if (!proxyKey) {
    res.status(500).json({ error: { message: "PROXY_API_KEY not configured", type: "server_error" } });
    return false;
  }

  // Accept both Authorization: Bearer <key> (OpenAI SDK style)
  // and X-Api-Key: <key> (Anthropic SDK / Claude Code CLI style)
  let token: string | undefined;
  const auth = req.headers.authorization;
  if (auth && auth.startsWith("Bearer ")) {
    token = auth.slice(7);
  } else if (req.headers["x-api-key"]) {
    token = req.headers["x-api-key"] as string;
  }

  if (!token) {
    res.status(401).json({ error: { message: "Missing API key (use Authorization: Bearer or X-Api-Key)", type: "invalid_request_error" } });
    return false;
  }
  if (token !== proxyKey) {
    res.status(401).json({ error: { message: "Invalid API key", type: "invalid_request_error" } });
    return false;
  }
  return true;
}

function isOpenAIModel(model: string): boolean {
  return model.startsWith("gpt-") || model.startsWith("o") && !model.startsWith("other");
}

function isAnthropicModel(model: string): boolean {
  return model.startsWith("claude-");
}

// Convert OpenAI tool format to Anthropic tool format
function openaiToolsToAnthropic(tools: OpenAI.Chat.ChatCompletionTool[]): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
    input_schema: (t.function.parameters ?? {}) as Anthropic.Tool.InputSchema,
  }));
}

// Convert OpenAI tool_choice to Anthropic tool_choice
function openaiToolChoiceToAnthropic(
  toolChoice: OpenAI.Chat.ChatCompletionToolChoiceOption | undefined
): Anthropic.MessageCreateParams["tool_choice"] | undefined {
  if (!toolChoice) return undefined;
  if (toolChoice === "none") return undefined;
  if (toolChoice === "auto") return { type: "auto" };
  if (toolChoice === "required") return { type: "any" };
  if (typeof toolChoice === "object" && toolChoice.type === "function") {
    return { type: "tool", name: toolChoice.function.name };
  }
  return undefined;
}

// Convert OpenAI messages to Anthropic messages format
function openaiMessagesToAnthropic(
  messages: OpenAI.Chat.ChatCompletionMessageParam[]
): { system?: string; messages: Anthropic.MessageParam[] } {
  let system: string | undefined;
  const anthropicMessages: Anthropic.MessageParam[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      system = typeof msg.content === "string" ? msg.content : "";
      continue;
    }

    if (msg.role === "user") {
      anthropicMessages.push({
        role: "user",
        content: typeof msg.content === "string" ? msg.content : (msg.content as Anthropic.ContentBlockParam[]),
      });
      continue;
    }

    if (msg.role === "assistant") {
      const content: Anthropic.ContentBlock[] = [];
      if (msg.content) {
        content.push({ type: "text", text: typeof msg.content === "string" ? msg.content : "" });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let parsedInput: Record<string, unknown> = {};
          try {
            parsedInput = JSON.parse(tc.function.arguments ?? "{}") as Record<string, unknown>;
          } catch {
            parsedInput = {};
          }
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: parsedInput,
          });
        }
      }
      anthropicMessages.push({ role: "assistant", content });
      continue;
    }

    if (msg.role === "tool") {
      const lastMsg = anthropicMessages[anthropicMessages.length - 1];
      const toolResultBlock: Anthropic.ToolResultBlockParam = {
        type: "tool_result",
        tool_use_id: msg.tool_call_id ?? "",
        content: typeof msg.content === "string" ? msg.content : "",
      };
      if (lastMsg && lastMsg.role === "user" && Array.isArray(lastMsg.content)) {
        (lastMsg.content as Anthropic.ContentBlockParam[]).push(toolResultBlock);
      } else {
        anthropicMessages.push({ role: "user", content: [toolResultBlock] });
      }
    }
  }

  return { system, messages: anthropicMessages };
}

// Convert Anthropic response to OpenAI format
function anthropicToOpenAI(msg: Anthropic.Message): OpenAI.Chat.ChatCompletion {
  const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = [];
  let textContent = "";

  for (const block of msg.content) {
    if (block.type === "text") {
      textContent += block.text;
    } else if (block.type === "tool_use") {
      toolCalls.push({
        id: block.id,
        type: "function",
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input),
        },
      });
    }
  }

  const finishReason: OpenAI.Chat.ChatCompletion.Choice["finish_reason"] =
    msg.stop_reason === "tool_use" ? "tool_calls" :
    msg.stop_reason === "end_turn" ? "stop" :
    msg.stop_reason === "max_tokens" ? "length" : "stop";

  const message: OpenAI.Chat.ChatCompletionMessage = {
    role: "assistant",
    content: textContent || null,
    refusal: null,
  };

  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  return {
    id: msg.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: msg.model,
    choices: [{
      index: 0,
      message,
      finish_reason: finishReason,
      logprobs: null,
    }],
    usage: {
      prompt_tokens: msg.usage.input_tokens,
      completion_tokens: msg.usage.output_tokens,
      total_tokens: msg.usage.input_tokens + msg.usage.output_tokens,
    },
  };
}

// ─── GET /v1/models ──────────────────────────────────────────────────────────

router.get("/models", (req, res) => {
  if (!verifyToken(req, res)) return;

  const now = Math.floor(Date.now() / 1000);
  const models = [
    ...OPENAI_MODELS.map((id) => ({ id, object: "model", created: now, owned_by: "openai" })),
    ...ANTHROPIC_MODELS.map((id) => ({ id, object: "model", created: now, owned_by: "anthropic" })),
  ];

  res.json({ object: "list", data: models });
});

// ─── POST /v1/chat/completions ────────────────────────────────────────────────

router.post("/chat/completions", async (req: Request, res: Response) => {
  if (!verifyToken(req, res)) return;

  const body = req.body as {
    model: string;
    messages: OpenAI.Chat.ChatCompletionMessageParam[];
    stream?: boolean;
    tools?: OpenAI.Chat.ChatCompletionTool[];
    tool_choice?: OpenAI.Chat.ChatCompletionToolChoiceOption;
    max_tokens?: number;
    temperature?: number;
    [key: string]: unknown;
  };

  const { model, messages, stream, tools, tool_choice, max_tokens, ...rest } = body;

  if (!model) {
    res.status(400).json({ error: { message: "model is required", type: "invalid_request_error" } });
    return;
  }

  try {
    // ── OpenAI routing ──
    if (isOpenAIModel(model)) {
      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          res.write(": keepalive\n\n");
          res.flush?.();
        }, 5000);

        req.on("close", () => clearInterval(keepalive));

        try {
          const params: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
            model,
            messages,
            stream: true,
            ...(tools ? { tools } : {}),
            ...(tool_choice ? { tool_choice } : {}),
            ...(max_tokens ? { max_completion_tokens: max_tokens } : {}),
            ...rest,
          } as OpenAI.Chat.ChatCompletionCreateParamsStreaming;

          const s = await openaiClient.chat.completions.create(params);
          for await (const chunk of s) {
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            res.flush?.();
          }
          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          res.end();
        }
      } else {
        const params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
          model,
          messages,
          stream: false,
          ...(tools ? { tools } : {}),
          ...(tool_choice ? { tool_choice } : {}),
          ...(max_tokens ? { max_completion_tokens: max_tokens } : {}),
          ...rest,
        } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming;

        const result = await openaiClient.chat.completions.create(params);
        res.json(result);
      }
      return;
    }

    // ── Anthropic routing ──
    if (isAnthropicModel(model)) {
      const { system, messages: anthropicMessages } = openaiMessagesToAnthropic(messages);
      const anthropicTools = tools ? openaiToolsToAnthropic(tools) : undefined;
      const anthropicToolChoice = openaiToolChoiceToAnthropic(tool_choice);

      const createParams: Anthropic.MessageCreateParamsNonStreaming = {
        model,
        max_tokens: max_tokens ?? 8192,
        messages: anthropicMessages,
        ...(system ? { system } : {}),
        ...(anthropicTools ? { tools: anthropicTools } : {}),
        ...(anthropicToolChoice ? { tool_choice: anthropicToolChoice } : {}),
      };

      // Forward anthropic-beta from client to upstream
      const anthropicBetaCC = req.headers["anthropic-beta"] as string | undefined;
      const sdkOptsCC = anthropicBetaCC
        ? { headers: { "anthropic-beta": anthropicBetaCC } }
        : {};

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          if (!res.writableEnded) {
            res.write(": keepalive\n\n");
            res.flush?.();
          }
        }, 5000);

        req.on("close", () => clearInterval(keepalive));

        try {
          // Stream Anthropic and convert to OpenAI chunk format
          const streamParams = { ...createParams, stream: true } as Anthropic.MessageStreamParams;
          const anthropicStream = anthropicClient.messages.stream(streamParams, sdkOptsCC);

          let messageId = `chatcmpl-${Date.now()}`;
          let currentToolCallIndex = 0;
          const toolCallAccumulator: Map<number, { id: string; name: string; args: string }> = new Map();

          for await (const event of anthropicStream) {
            if (event.type === "message_start") {
              messageId = event.message.id;
              // Send initial chunk
              const initChunk: OpenAI.Chat.ChatCompletionChunk = {
                id: messageId,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null, logprobs: null }],
              };
              res.write(`data: ${JSON.stringify(initChunk)}\n\n`);
              res.flush?.();
            } else if (event.type === "content_block_start") {
              if (event.content_block.type === "tool_use") {
                const toolBlock = event.content_block as Anthropic.ToolUseBlock;
                toolCallAccumulator.set(event.index, { id: toolBlock.id, name: toolBlock.name, args: "" });
                const chunk: OpenAI.Chat.ChatCompletionChunk = {
                  id: messageId,
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model,
                  choices: [{
                    index: 0,
                    delta: {
                      tool_calls: [{
                        index: currentToolCallIndex,
                        id: toolBlock.id,
                        type: "function",
                        function: { name: toolBlock.name, arguments: "" },
                      }],
                    },
                    finish_reason: null,
                    logprobs: null,
                  }],
                };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                res.flush?.();
                currentToolCallIndex++;
              }
            } else if (event.type === "content_block_delta") {
              if (event.delta.type === "text_delta") {
                const chunk: OpenAI.Chat.ChatCompletionChunk = {
                  id: messageId,
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model,
                  choices: [{ index: 0, delta: { content: event.delta.text }, finish_reason: null, logprobs: null }],
                };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                res.flush?.();
              } else if (event.delta.type === "input_json_delta") {
                const acc = toolCallAccumulator.get(event.index);
                if (acc) {
                  acc.args += event.delta.partial_json;
                  const tcIndex = event.index;
                  const chunk: OpenAI.Chat.ChatCompletionChunk = {
                    id: messageId,
                    object: "chat.completion.chunk",
                    created: Math.floor(Date.now() / 1000),
                    model,
                    choices: [{
                      index: 0,
                      delta: {
                        tool_calls: [{
                          index: tcIndex,
                          function: { arguments: event.delta.partial_json },
                        }],
                      },
                      finish_reason: null,
                      logprobs: null,
                    }],
                  };
                  res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                  res.flush?.();
                }
              }
            } else if (event.type === "message_delta") {
              const finishReason: OpenAI.Chat.ChatCompletionChunk.Choice["finish_reason"] =
                event.delta.stop_reason === "tool_use" ? "tool_calls" :
                event.delta.stop_reason === "end_turn" ? "stop" :
                event.delta.stop_reason === "max_tokens" ? "length" : "stop";

              const chunk: OpenAI.Chat.ChatCompletionChunk = {
                id: messageId,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [{ index: 0, delta: {}, finish_reason: finishReason, logprobs: null }],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              res.flush?.();
            }
          }

          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          if (!res.writableEnded) res.end();
        }
      } else {
        // Non-streaming: use stream().finalMessage() to avoid long timeout
        const anthropicStream = anthropicClient.messages.stream(
          createParams as Anthropic.MessageStreamParams,
          sdkOptsCC,
        );
        const finalMsg = await anthropicStream.finalMessage();
        const openaiResponse = anthropicToOpenAI(finalMsg);
        res.json(openaiResponse);
      }
      return;
    }

    res.status(400).json({ error: { message: `Unknown model: ${model}`, type: "invalid_request_error" } });
  } catch (err) {
    logger.error({ err }, "Proxy error in /v1/chat/completions");
    if (!res.headersSent) {
      res.status(500).json({ error: { message: "Internal proxy error", type: "server_error" } });
    } else if (!res.writableEnded) {
      res.write(`data: ${JSON.stringify({ error: { message: "Stream error", type: "server_error" } })}\n\n`);
      res.end();
    }
  }
});

// Normalize system: accept string or array of text blocks
function normalizeSystem(
  system: unknown
): Anthropic.MessageCreateParams["system"] | undefined {
  if (!system) return undefined;
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    // Flatten array of text blocks into a single string for simplicity,
    // or pass as-is if the SDK supports typed array (it does as of recent versions)
    const blocks = system as Array<{ type: string; text?: string }>;
    return blocks
      .filter((b) => b.type === "text" && b.text)
      .map((b) => b.text!)
      .join("\n\n");
  }
  return undefined;
}

// ─── POST /v1/messages (Anthropic native) ──────────────────────────────────────

router.post("/messages", async (req: Request, res: Response) => {
  if (!verifyToken(req, res)) return;

  const body = req.body as {
    model: string;
    messages: Anthropic.MessageParam[];
    system?: unknown;
    tools?: Anthropic.Tool[];
    tool_choice?: Anthropic.MessageCreateParams["tool_choice"];
    max_tokens?: number;
    stream?: boolean;
    metadata?: Anthropic.MessageCreateParams["metadata"];
    temperature?: number;
    thinking?: { type: "enabled"; budget_tokens: number };
    top_k?: number;
    top_p?: number;
    stop_sequences?: string[];
    [key: string]: unknown;
  };

  const {
    model, messages, tools, tool_choice, max_tokens, stream,
    metadata, temperature, thinking, top_k, top_p, stop_sequences,
  } = body;
  const system = normalizeSystem(body.system);

  // Forward anthropic-beta header from client to upstream Anthropic
  const anthropicBeta = req.headers["anthropic-beta"] as string | undefined;

  if (!model) {
    res.status(400).json({ error: { message: "model is required" } });
    return;
  }

  try {
    // ── Claude native passthrough ──
    if (isAnthropicModel(model)) {
      // Only pass known-valid Anthropic fields to avoid 400s from unknown keys
      const createParams: Anthropic.MessageCreateParams = {
        model,
        messages,
        max_tokens: max_tokens ?? 8192,
        ...(system ? { system } : {}),
        ...(tools && tools.length > 0 ? { tools } : {}),
        ...(tool_choice ? { tool_choice } : {}),
        ...(metadata ? { metadata } : {}),
        ...(temperature !== undefined ? { temperature } : {}),
        ...(thinking ? { thinking } : {}),
        ...(top_k !== undefined ? { top_k } : {}),
        ...(top_p !== undefined ? { top_p } : {}),
        ...(stop_sequences ? { stop_sequences } : {}),
      };

      // Extra SDK request options — forward anthropic-beta if present
      const sdkOptions = anthropicBeta
        ? { headers: { "anthropic-beta": anthropicBeta } }
        : {};

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          if (!res.writableEnded) {
            res.write(": keepalive\n\n");
            res.flush?.();
          }
        }, 5000);

        req.on("close", () => clearInterval(keepalive));

        try {
          const anthropicStream = anthropicClient.messages.stream(
            createParams as Anthropic.MessageStreamParams,
            sdkOptions,
          );
          for await (const event of anthropicStream) {
            if (res.writableEnded) break;
            res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
            res.flush?.();
          }
        } finally {
          clearInterval(keepalive);
          if (!res.writableEnded) res.end();
        }
      } else {
        const anthropicStream = anthropicClient.messages.stream(
          createParams as Anthropic.MessageStreamParams,
          sdkOptions,
        );
        const finalMsg = await anthropicStream.finalMessage();
        res.json(finalMsg);
      }
      return;
    }

    // ── OpenAI model via Anthropic /v1/messages interface ──
    if (isOpenAIModel(model)) {
      // Convert Anthropic messages to OpenAI format
      const openaiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

      if (system) {
        openaiMessages.push({ role: "system", content: typeof system === "string" ? system : String(system) });
      }

      for (const msg of messages) {
        if (msg.role === "user") {
          const content = msg.content;
          if (typeof content === "string") {
            openaiMessages.push({ role: "user", content });
          } else {
            // Convert tool_result blocks back to tool role messages
            const toolResults: OpenAI.Chat.ChatCompletionToolMessageParam[] = [];
            const textParts: string[] = [];
            for (const block of content as Anthropic.ContentBlockParam[]) {
              if (block.type === "tool_result") {
                const tr = block as Anthropic.ToolResultBlockParam;
                toolResults.push({
                  role: "tool",
                  tool_call_id: tr.tool_use_id,
                  content: typeof tr.content === "string" ? tr.content : JSON.stringify(tr.content),
                });
              } else if (block.type === "text") {
                textParts.push((block as Anthropic.TextBlockParam).text);
              }
            }
            if (textParts.length > 0) {
              openaiMessages.push({ role: "user", content: textParts.join("\n") });
            }
            openaiMessages.push(...toolResults);
          }
        } else if (msg.role === "assistant") {
          const content = msg.content;
          if (typeof content === "string") {
            openaiMessages.push({ role: "assistant", content });
          } else {
            const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = [];
            let textContent = "";
            for (const block of content as Anthropic.ContentBlock[]) {
              if (block.type === "text") textContent += block.text;
              if (block.type === "tool_use") {
                toolCalls.push({
                  id: block.id,
                  type: "function",
                  function: { name: block.name, arguments: JSON.stringify(block.input) },
                });
              }
            }
            const assistantMsg: OpenAI.Chat.ChatCompletionAssistantMessageParam = {
              role: "assistant",
              content: textContent || null,
            };
            if (toolCalls.length > 0) assistantMsg.tool_calls = toolCalls;
            openaiMessages.push(assistantMsg);
          }
        }
      }

      // Convert Anthropic tools to OpenAI format
      let openaiTools: OpenAI.Chat.ChatCompletionTool[] | undefined;
      if (tools) {
        openaiTools = tools.map((t) => ({
          type: "function" as const,
          function: {
            name: t.name,
            description: t.description,
            parameters: t.input_schema as Record<string, unknown>,
          },
        }));
      }

      // Convert Anthropic tool_choice to OpenAI
      let openaiToolChoice: OpenAI.Chat.ChatCompletionToolChoiceOption | undefined;
      if (tool_choice) {
        if (tool_choice.type === "auto") openaiToolChoice = "auto";
        else if (tool_choice.type === "any") openaiToolChoice = "required";
        else if (tool_choice.type === "tool") {
          openaiToolChoice = { type: "function", function: { name: (tool_choice as Anthropic.ToolChoiceTool).name } };
        }
      }

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          if (!res.writableEnded) {
            res.write(": keepalive\n\n");
            res.flush?.();
          }
        }, 5000);

        req.on("close", () => clearInterval(keepalive));

        try {
          const oaiStream = await openaiClient.chat.completions.create({
            model,
            messages: openaiMessages,
            stream: true,
            ...(openaiTools ? { tools: openaiTools } : {}),
            ...(openaiToolChoice ? { tool_choice: openaiToolChoice } : {}),
            ...(max_tokens ? { max_completion_tokens: max_tokens } : {}),
          });

          // Synthesize Anthropic SSE events from OpenAI stream
          const msgId = `msg_${Date.now()}`;
          const created = Math.floor(Date.now() / 1000);

          res.write(`event: message_start\ndata: ${JSON.stringify({
            type: "message_start",
            message: { id: msgId, type: "message", role: "assistant", content: [], model, stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } }
          })}\n\n`);
          res.flush?.();

          res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`);
          res.flush?.();

          let toolCallBlocks: Map<string, { index: number; name: string; args: string }> = new Map();
          let textBlockOpen = true;
          let textBlockIndex = 0;
          let nextBlockIndex = 1;
          let finishReason = "end_turn";
          let outputTokens = 0;

          for await (const chunk of oaiStream) {
            const delta = chunk.choices[0]?.delta;
            if (!delta) continue;

            if (delta.content) {
              if (!textBlockOpen) {
                res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: textBlockIndex, content_block: { type: "text", text: "" } })}\n\n`);
                res.flush?.();
                textBlockOpen = true;
              }
              res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: textBlockIndex, delta: { type: "text_delta", text: delta.content } })}\n\n`);
              res.flush?.();
              outputTokens++;
            }

            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                if (tc.id) {
                  // New tool call block
                  if (textBlockOpen) {
                    res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: textBlockIndex })}\n\n`);
                    res.flush?.();
                    textBlockOpen = false;
                  }
                  const blockIdx = nextBlockIndex++;
                  toolCallBlocks.set(tc.id, { index: blockIdx, name: tc.function?.name ?? "", args: "" });
                  res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: blockIdx, content_block: { type: "tool_use", id: tc.id, name: tc.function?.name ?? "", input: {} } })}\n\n`);
                  res.flush?.();
                }
                if (tc.function?.arguments) {
                  // Find the block by iterating (streaming ID matching)
                  for (const [, blk] of toolCallBlocks) {
                    if (blk.index === (nextBlockIndex - 1) || tc.index !== undefined) {
                      // Use tc.index to find the right block
                      const matchingEntry = Array.from(toolCallBlocks.entries()).find((_, idx) => idx === tc.index);
                      const entry = matchingEntry ? matchingEntry[1] : Array.from(toolCallBlocks.values()).at(-1);
                      if (entry) {
                        entry.args += tc.function.arguments;
                        res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: entry.index, delta: { type: "input_json_delta", partial_json: tc.function.arguments } })}\n\n`);
                        res.flush?.();
                      }
                      break;
                    }
                  }
                }
              }
            }

            const fr = chunk.choices[0]?.finish_reason;
            if (fr === "tool_calls") finishReason = "tool_use";
            else if (fr === "length") finishReason = "max_tokens";
            else if (fr === "stop") finishReason = "end_turn";

            if (chunk.usage) outputTokens = chunk.usage.completion_tokens ?? outputTokens;
          }

          // Close open blocks
          if (textBlockOpen) {
            res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: textBlockIndex })}\n\n`);
            res.flush?.();
          }
          for (const blk of toolCallBlocks.values()) {
            res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: blk.index })}\n\n`);
            res.flush?.();
          }

          res.write(`event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: finishReason, stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`);
          res.flush?.();
          res.write(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
          res.flush?.();
        } finally {
          clearInterval(keepalive);
          if (!res.writableEnded) res.end();
        }
      } else {
        // Non-streaming: call OpenAI and convert to Anthropic Message format
        const oaiResult = await openaiClient.chat.completions.create({
          model,
          messages: openaiMessages,
          stream: false,
          ...(openaiTools ? { tools: openaiTools } : {}),
          ...(openaiToolChoice ? { tool_choice: openaiToolChoice } : {}),
          ...(max_tokens ? { max_completion_tokens: max_tokens } : {}),
        });

        const choice = oaiResult.choices[0];
        const content: Anthropic.ContentBlock[] = [];

        if (choice.message.content) {
          content.push({ type: "text", text: choice.message.content });
        }

        if (choice.message.tool_calls) {
          for (const tc of choice.message.tool_calls) {
            let parsedInput: Record<string, unknown> = {};
            try { parsedInput = JSON.parse(tc.function.arguments ?? "{}") as Record<string, unknown>; } catch { /* */ }
            content.push({ type: "tool_use", id: tc.id, name: tc.function.name, input: parsedInput });
          }
        }

        const stopReason: Anthropic.Message["stop_reason"] =
          choice.finish_reason === "tool_calls" ? "tool_use" :
          choice.finish_reason === "length" ? "max_tokens" : "end_turn";

        const anthropicResponse: Anthropic.Message = {
          id: oaiResult.id,
          type: "message",
          role: "assistant",
          content,
          model: oaiResult.model,
          stop_reason: stopReason,
          stop_sequence: null,
          usage: {
            input_tokens: oaiResult.usage?.prompt_tokens ?? 0,
            output_tokens: oaiResult.usage?.completion_tokens ?? 0,
          },
        };

        res.json(anthropicResponse);
      }
      return;
    }

    res.status(400).json({ error: { message: `Unknown model: ${model}` } });
  } catch (err) {
    logger.error({ err }, "Proxy error in /v1/messages");
    if (!res.headersSent) {
      res.status(500).json({ error: { message: "Internal proxy error" } });
    } else if (!res.writableEnded) {
      res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { message: "Stream error" } })}\n\n`);
      res.end();
    }
  }
});

export default router;
