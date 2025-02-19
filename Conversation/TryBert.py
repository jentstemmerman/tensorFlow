from transformers import pipeline

# Load a Question Answering model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Example conversation
context = "ChatGPT is an AI model developed by OpenAI. It can assist users in answering questions and providing information. A cat is a domestic animal that is often kept as a pet. Cats are known for their playful behavior and friendly nature."

question = "Who developed ChatGPT?"
answer = qa_pipeline(question=question, context=context)
print("Answer:", answer["answer"])


question = "What can ChatGPT do?"
answer = qa_pipeline(question=question, context=context)
print("Answer:", answer["answer"])

context = "hello"
question = "What is a cat?"
answer = qa_pipeline(question=question, context=context)
print("Answer:", answer["answer"])