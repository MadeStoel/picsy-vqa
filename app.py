from flask import Flask, request, jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load the processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa")


@app.route('/vqa', methods=['POST'])
def vqa():
    file = request.files['image']
    text = request.form['question']
    image = Image.open(io.BytesIO(file.read()))

    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return jsonify({'answer': model.config.id2label[idx]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
