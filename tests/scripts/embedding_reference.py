"""Asks running llama-servers for the EmbeddingGemma 2 reference vectors.

Run by embedding_reference.sh, which builds the server and starts one for the
projector's default image budget and one for each budget the script lists. Each
input goes through /embeddings as `content`, so the server tokenizes it with
the model's special tokens (BOS and EOS) and pools it whole, the way the
native embedding input must reproduce. Standard library only.

    python3 -I embedding_reference.py <fixtures-dir> <out.json> <server-flags> <url> [<budget>=<url> ...]
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

# A recording, with the query that must find it among the images and the clip.
AUDIO = ("asr-counting.wav", "a voice counting from one to five, then saying when the meeting is")

# Inputs that mix text and media, as ordered parts: the cases that need the stream whole.
MIXED = [
    ("text and image", [{"text": "title: none | text: a photo of "}, {"image": "cat.jpg"},
                        {"text": " sitting on a mat"}]),
    ("text, image, video and audio", [{"text": "title: none | text: "}, {"image": "cat.jpg"},
                                      {"text": " beside "}, {"video": "video/red-square-moves.mp4"},
                                      {"text": " while a voice says "}, {"audio": "asr-counting.wav"}]),
]

# Every task framing on the model card, and a titled document, as the model card writes them.
TASKS = [
    ("question answering", "task: question answering | query: What animal is sitting in the photo?"),
    ("fact checking", "task: fact checking | query: Robots assemble car bodies on a factory line."),
    ("code retrieval", "task: code retrieval | query: a function that adds two numbers"),
    ("classification", "task: classification | query: The striker scored in the final minute of the match."),
    ("clustering", "task: clustering | query: Quarterly revenue fell because of supply delays."),
    ("sentence similarity", "task: sentence similarity | query: A cat is resting on a rug."),
    ("titled document", "title: add.py | text: def add(a, b):\n    return a + b"),
]

# Queries in other languages, with the image each must find.
MULTILINGUAL = [
    ("una foto de un gato", "cat.jpg"),
    ("工場の組立ラインで車体を組み立てるロボット", "embedding/navvis-factory-planning2-image1.jpg"),
]

# The image embedded at the projector's smallest and largest image budgets, each by its own server.
BUDGET_IMAGE = "embedding/f1.jpg"

# A video, read by llama.cpp's own video helper at its default 4 frames a
# second when its file sits behind a marker: the frames committed beside it. Embedding the frames natively must
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


def file_bytes(fixtures, name):
    with open(f"{fixtures}/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode()


def props_of(url):
    return json.load(urllib.request.urlopen(f"{url}/props"))


def main(fixtures, out, server, url, *budgeted):
    props = props_of(url)

    def parts(ordered, marker=props["media_marker"]):
        """Ordered parts as one prompt: text as written, each media file behind the server's own marker, in order."""
        media = [part for part in ordered if "text" not in part]
        return {"prompt_string": "".join(part.get("text", marker) for part in ordered),
                "multimodal_data": [file_bytes(fixtures, next(iter(part.values()))) for part in media]}

    def query(text, finds):
        return {"name": f"query: {text}", "kind": "text", "text": QUERY + text, "finds": finds,
                "vector": vector(url, QUERY + text)}

    entries = []
    for name, text in IMAGES:
        entries.append({"name": name, "kind": "image", "image": name, "vector": vector(url, parts([{"image": name}]))})
        entries.append(query(text, name))
    clip, text = VIDEO
    entries.append({"name": clip, "kind": "video", "video": clip, "vector": vector(url, parts([{"video": clip}]))})
    entries.append(query(text, clip))
    recording, text = AUDIO
    entries.append({"name": recording, "kind": "audio", "audio": recording,
                    "vector": vector(url, parts([{"audio": recording}]))})
    entries.append(query(text, recording))
    for name, ordered in MIXED:
        entries.append({"name": name, "kind": "mixed", "parts": ordered, "vector": vector(url, parts(ordered))})
    for task, text in TASKS:
        entries.append({"name": task, "kind": "text", "task": task, "text": text, "vector": vector(url, text)})
    for text, finds in MULTILINGUAL:
        entries.append(query(text, finds))
    for budget, budget_url in (pair.split("=", 1) for pair in budgeted):
        entries.append({"name": f"{BUDGET_IMAGE} at {budget} image tokens", "kind": "image", "image": BUDGET_IMAGE,
                        "image_tokens": int(budget),
                        "vector": vector(budget_url, parts([{"image": BUDGET_IMAGE}], props_of(budget_url)["media_marker"]))})

    reference = {
        "model": os.path.basename(props.get("model_path", "")),
        "server": f"{server}, content with special tokens, L2-normalised; an entry with image_tokens came from a "
                  "server run with --image-min-tokens and --image-max-tokens both at that budget",
        "dimension": len(entries[0]["vector"]),
        "entries": entries,
    }
    with open(out, "w") as f:
        json.dump(reference, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
