// Pulls the REBUILT leaderboard (solo, duos, trios) over the public websocket.
// Usage: node frsbot/scrape/fetch_board.mjs data/board.json
import { writeFileSync } from "node:fs";

const url = "wss://reefscape-ws.frsim.com/rebuilt-ws";
const out = process.argv[2] ?? "data/board.json";

const ws = new WebSocket(url);
const timer = setTimeout(() => fail("timed out waiting for board"), 10_000);
const fail = (msg) => { console.error(msg); process.exit(1); };

ws.onopen = () => ws.send(JSON.stringify({ t: "rebuiltBoard" }));
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  if (d.t !== "board") return;
  clearTimeout(timer);
  writeFileSync(out, JSON.stringify(d));
  console.log(`wrote ${out}: entries=${d.entries.length} duos=${d.duos.length} trios=${d.trios.length}`);
  ws.close();
};
ws.onerror = () => fail("websocket error");
