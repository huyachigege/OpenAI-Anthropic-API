import http from "http";
import app from "./app";
import { logger } from "./lib/logger";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const server = http.createServer(app);

// Disable Node.js default requestTimeout (300 s) so that long-running
// extended-thinking requests (which can exceed 5 minutes) are never
// forcibly terminated by the HTTP layer.  Timeout enforcement is left
// to the Anthropic/OpenAI SDK and the upstream platform.
server.requestTimeout = 0;

// headersTimeout only covers the time to receive the incoming request
// headers, not the response stream.  Keep it at a reasonable value.
server.headersTimeout = 30_000;

server.listen(port, (err?: Error) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info(
    { port, requestTimeout: server.requestTimeout, headersTimeout: server.headersTimeout },
    "Server listening",
  );
});
