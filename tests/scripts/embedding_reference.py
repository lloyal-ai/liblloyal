"""Asks a running llama-server for the EmbeddingGemma 2 reference vectors.

Run by embedding_reference.sh, which builds the server and starts it. Each
input goes through /embeddings as `content`, so the server tokenizes it with
the model's special tokens (BOS and EOS) and pools it whole, the way the
native embedding input must reproduce. Standard library only.

    python3 -I embedding_reference.py <server-url> <fixtures-dir> <out.json>
"""

import base64
import json
import os
import sys
import urllib.request

QUERY = "task: search result | query: "

# Each image with the query that must find it.
IMAGES = [
    ("cat.jpg", "a photo of a cat"),
    ("embedding/f1.jpg", "an open-plan office with desks, screens and plants"),
    ("embedding/navvis-factory-planning2-image1.jpg", "robots assembling car bodies on a factory line"),
    ("embedding/Global-X-Digital-Evolution-Series-Aug-25-Part-2.jpg", "an infographic of AI adoption by industry"),
]

# Text and an image in one input: the case that needs the stream whole.
INTERLEAVED = ("title: none | text: a photo of ", "cat.jpg", " sitting on a mat")

# A video, read by llama.cpp's own video helper at its default 4 frames a
# second: the frames committed beside it. Embedding the frames natively must
# land where the server puts the clip.
VIDEO = ("video/red-square-moves.mp4", "a red square moving from left to right across a white frame")


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def vector(url, content):
    [result] = post(f"{url}/embeddings", {"content": [content]})
    [pooled] = result["embedding"]
    return pooled


def image_bytes(fixtures, name):
    with open(f"{fixtures}/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode()


def main(url, fixtures, out):
    props = json.load(urllib.request.urlopen(f"{url}/props"))
    marker = props["media_marker"]

    def media(text, name):
        return {"prompt_string": text, "multimodal_data": [image_bytes(fixtures, name)]}

    entries = []
    for name, query in IMAGES:
        entries.append({"name": name, "kind": "image", "image": name, "vector": vector(url, media(marker, name))})
        entries.append({"name": f"query: {query}", "kind": "text", "text": QUERY + query,
                        "finds": name, "vector": vector(url, QUERY + query)})
    clip, query = VIDEO
    entries.append({"name": clip, "kind": "video", "video": clip,
                    "vector": vector(url, {"content": [{"type": "input_video", "input_video": {
                        "data": "data:video/mp4;base64," + image_bytes(fixtures, clip)}}]})})
    entries.append({"name": f"query: {query}", "kind": "text", "text": QUERY + query,
                    "finds": clip, "vector": vector(url, QUERY + query)})
    before, name, after = INTERLEAVED
    entries.append({"name": "text and image", "kind": "text+image", "text_before": before, "image": name,
                    "text_after": after, "vector": vector(url, media(before + marker + after, name))})

    reference = {
        "model": os.path.basename(props.get("model_path", "")),
        "server": "llama-server --embeddings --pooling mean -c 8192 -b 8192 -ub 8192 -ngl 0, content with special tokens, L2-normalised",
        "dimension": len(entries[0]["vector"]),
        "entries": entries,
    }
    with open(out, "w") as f:
        json.dump(reference, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
