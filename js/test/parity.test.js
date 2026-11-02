// JS vs Python parity. The expected values are produced by scripts/make_fixtures.py with the Python reference implementation.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { sentFeats, TOKEN } = require("../src/features.js");
const { LayeredTagger } = require("../src/tagger.js");

const fx = (n) => JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", n), "utf8"));
const model = (n) => JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "models", n), "utf8"));
const tokenize = (t) => t.match(new RegExp(TOKEN.source, "gu")) || [];

test("tokenizer matches Python", () => {
  for (const e of fx("tokens.json")) assert.deepStrictEqual(tokenize(e.text), e.tokens, JSON.stringify(e.text));
});

test("features match Python, token by token", () => {
  for (const e of fx("features.json")) assert.deepStrictEqual(sentFeats(e.tokens), e.feats, JSON.stringify(e.tokens));
});

test("tags of both layers match Python", () => {
  const lt = new LayeredTagger(model("note.json"), model("cite.json"));
  for (const e of fx("tags.json")) {
    const got = lt.tagTokens(e.tokens);
    assert.deepStrictEqual(got.note, e.note, e.text);
    assert.deepStrictEqual(got.cite, e.cite, e.text);
  }
});

test("annotate returns notes (with kind) and cites with offsets, like Python", () => {
  const lt = new LayeredTagger(model("note.json"), model("cite.json"));
  for (const e of fx("tags.json")) assert.deepStrictEqual(lt.annotate(e.text), e.annotate, e.text);
});
