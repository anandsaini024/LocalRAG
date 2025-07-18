from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

model = ChatAnthropic(model = 'claude-3-5-sonnet-20241022')

paper_input = st.selectbox("Select Research Paper", ["Select...", "Attention is All you need", "BERT: Pre-training of Deep Bidirectional Transformers"])

style_input = st.selectbox("Select Explanation Style", ["Beginner Friendly", "Technical", "Code-Oriented", "Mathematics"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs", "Medium (3-5 Paragraphs", "Long (Detailed Explanation"])

# Template
template = PromptTemplate(
    template = """Please summarize the research paper titled {paper_input} with the following specifications: 
    Explanation Style: {style_input}  
    Explanation Length: {length_input}  
    1. Mathematical Details:    
                - Include relevant mathematical equations if present in the paper.
                - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:  
                - Use relatable analogies to simplify complex ideas. 
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing. Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
    input_variables = ['paper_input', 'style_input', 'length_input'],
    validate_template = True
)

# Fill in the placeholders

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)

