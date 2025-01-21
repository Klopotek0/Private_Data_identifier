from detect import detect_and_recognize_text
import matplotlib.pyplot as plt
import cv2

model_path = "yolo.pt"

image_path = 'Text_detection.v2i.yolov8/test/images/MD_O_png.rf.23d8a89e43c258bf759a7dc01c570134.jpg'
detected_texts, image = detect_and_recognize_text(image_path, model_path)

full_text = " ".join([item['text'] for item in detected_texts])


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = full_text
messages = [
    {"role": "system", "content": "You are expert of identifying private data in medical studies, you will receive one big string with text and will only respond with the data you consider private human data. This includes surnames, names, dates, places, IDs"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

private_data_list = response.split()  

for item in detected_texts:
    text = item['text'] 
    bbox = item['bbox']

    if any(private_word in text for private_word in private_data_list):
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=3)  
        print("cos jest")
    else:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 255, 0), thickness=2) 
        print(text)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


