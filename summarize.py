import json

def truncate_lists(obj, max_items=5):
    if isinstance(obj, dict):
        return {k: truncate_lists(v, max_items) for k, v in obj.items()}
    elif isinstance(obj, list):
        truncated = [truncate_lists(i, max_items) for i in obj[:max_items]]
        if len(obj) > max_items:
            truncated.append({"__truncated__": True})
        return truncated
    else:
        return obj

# Load your JSON
with open("output/1751220679_2007_res.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process it
condensed = truncate_lists(data)

# Save output
with open("condensed.json", "w", encoding="utf-8") as f:
    json.dump(condensed, f, indent=2, ensure_ascii=False)
