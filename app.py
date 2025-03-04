import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from decouple import config

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

st.set_page_config(
    page_title='Estoque GPT'
)

st.header('Assistente de Estoque')

model_options = [
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4o-mini',
    'gpt-40',
]

selected_model = st.sidebar.selectbox(
    label='Selecione o modelo LLM',
    options=model_options
)

st.sidebar.markdown("""
## Sobre

Este agente consulta um banco de dados de estoque utilizando um modelo GPT.
""")

st.write('Faca perguntas sobre o estoque de produtos, precos e reposicoes.')
user_question = st.text_input('O que deseja saber sobre o estoque?')



model = ChatOpenAI(
    model=selected_model
)

db = SQLDatabase.from_uri('sqlite:///estoque.db')

toolkit = SQLDatabaseToolkit(
    llm=model,
    db=db,
)

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
)

prompt = '''
Use as ferramentas necessárias para responder perguntas relacionadas ao estoque de produtos. 
Você fornecerá insights sobre produtos, precos, reposicao de estoque e relatórios conforme solicitado pelo usuário.
A resposta final deve ter uma formatacão amigável de visualizacão para o usuário.
Sempre responda em português brasilerio.
Pergunta: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)





if st.button('Consultar'):
    if user_question:
        formatted_prompt = prompt_template.format(q=user_question)
        output = agent_executor.invoke(
            {
                'input': formatted_prompt
            }
        )
        st.markdown(output.get('output'))
    else:
        st.warning('Por Favor, faca uma pergunta.')


