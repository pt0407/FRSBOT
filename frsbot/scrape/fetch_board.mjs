// Pulls the REBUILT leaderboard (solo, duos, trios) over the public websocket.
// Usage: node frsbot/scrape/fetch_board.mjs data/board.json
import { writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const url = "wss://reefscape-ws.frsim.com/rebuilt-ws";

export const queries = [{ t: "rebuiltBoard" }, ...["1v1", "2v2", "3v3"].map(mode => ({ t: "rankedBoard", mode }))];

export function queryBoard(query, Socket = WebSocket) {
  if (!queries.some(q => JSON.stringify(q) === JSON.stringify(query))) throw new Error("unsupported board query");
  return new Promise((resolve, reject) => {
    const ws = new Socket(url);
    let done = false;
    const finish = (error, result) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      ws.close();
      error ? reject(error) : resolve(result);
    };
    const timer = setTimeout(() => finish(new Error("timed out waiting for board")), 10000);
    ws.onopen = () => ws.send(JSON.stringify(query));
    ws.onmessage = event => {
      try {
        const data = JSON.parse(event.data);
        if (["error", "rankedError"].includes(data.t)) return finish(new Error(data.msg));
        if (query.t === "rebuiltBoard" ? data.t === "board" && data.game === "rebuilt"
          : data.t === "rankedBoard" && data.mode === query.mode) finish(null, data);
      } catch (error) { finish(error); }
    };
    ws.onerror = () => finish(new Error("websocket error; not retrying"));
    ws.onclose = () => finish(new Error("websocket closed before board"));
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const out = args.find(arg => !arg.startsWith("--")) ?? "data/board.json";
  try {
    if (args.includes("--all")) {
      const responses = [];
      for (const query of queries) {
        if (responses.length) await delay(1000);
        const data = await queryBoard(query);
        responses.push({ query, at: Date.now(), data });
        writeFileSync(out, JSON.stringify({ responses }, null, 1));
        console.log(`${JSON.stringify(query)} entries=${data.entries?.length ?? data.rows?.length} duos=${data.duos?.length ?? 0} trios=${data.trios?.length ?? 0}`);
      }
    } else {
      const data = await queryBoard(queries[0]);
      writeFileSync(out, JSON.stringify(data));
      console.log(`wrote ${out}: entries=${data.entries.length} duos=${data.duos.length} trios=${data.trios.length}`);
    }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
