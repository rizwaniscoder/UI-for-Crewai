# Import necessary libraries
import streamlit as st
import sys
import re
import PyPDF2
# Assuming these previously imported modules ("crewai", "langchain_openai", etc.) exist and are accurate based on the provided code snippet
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import os
from langchain_community.tools import DuckDuckGoSearchRun

duckduckgo_search = DuckDuckGoSearchRun()


# Improved StreamToExpander class which includes error handling and stream flushing
class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit  # To prevent memory overflow with large outputs

    def write(self, data):
        # Using regular expressions to remove ANSI codes that may clutter output
        cleaned_data = re.sub(r'\x1B\[\d+;?\d*m', '', data)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.pop(0)  # Remove the oldest entry when buffer limit is reached
        self.buffer.append(cleaned_data)

        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()  # Clear the buffer after printing to the expander

    def flush(self):
        if self.buffer:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

# Encapsulate the app's functionality within a class for better organization
class CrewAIApp:
    def __init__(self):
        # Initializing the LLM options as a class variable for easier modification and access
        self.llm_options = ['OpenAI GPT-4', 'Claude-3', 'Groq']

    def run(self):
        st.title("Mr. Slothuus - Crewai Agents Builder")

        # LLM configuration
        llm_option = st.selectbox("Choose LLM for Agents:", self.llm_options)
        api_key = st.text_input("Enter API Key for chosen LLM:", type="password")

        # Agent configuration
        number_of_agents = st.number_input("Number of Agents to Create:", min_value=1, max_value=10, value=1, step=1)
        agent_details = self.collect_agent_details(number_of_agents)

        # PDF Analysis
        pdf = st.file_uploader("Upload PDF for Analysis", type=['pdf'])

        # Task configuration
        tasks_list = [st.text_input(f"Task for Agent {i+1}", key=f"task_{i}") for i in range(number_of_agents)]

        # Execution with improved error validation
        if st.button("Start Crew Execution"):
            if not api_key:
                st.error("API Key is required.")
            elif not agent_details or not all(tasks_list):
                st.error("All agent details and tasks are required.")
            else:
                self.run_crew_analysis(agent_details, tasks_list, llm_option, api_key, pdf)

    def collect_agent_details(self, number_of_agents):
        agent_details = []
        for i in range(number_of_agents):
            with st.expander(f"Agent {i+1} Details"):
                role = st.text_input(f"Role for Agent {i+1}", key=f"role_{i}")
                goal = st.text_area(f"Goal for Agent {i+1}", key=f"goal_{i}")
                backstory = st.text_area(f"Backstory for Agent {i+1}", key=f"backstory_{i}")
                agent_details.append((role, goal, backstory))
        return agent_details

    def run_crew_analysis(self, agent_details, tasks_list, llm_option, api_key, pdf):
        process_output_expander = st.expander("Processing Output:")
        sys.stdout = StreamToExpander(process_output_expander)

        try:
            llm = self.setup_llm(llm_option, api_key)
            agents, tasks = self.initialize_agents_and_tasks(agent_details, tasks_list, llm)

            if not agents or not tasks:
                st.error("Failed to initialize agents or tasks.")
                return

            # Append PDF contents to the first agent's task description
            if pdf is not None and tasks:
                try:
                    pdf_reader = PyPDF2.PdfReader(pdf)
                    pdf_text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num] 
                        pdf_text += page.extract_text()
                    tasks[0].description += "\n\n" + pdf_text
                except Exception as e:
                    st.error(f"Failed to read PDF contents: {e}")

            crew = Crew(agents=agents, tasks=tasks, verbose=2, process=Process.sequential)
            crew_result = crew.kickoff()

            st.write(crew_result)
        except Exception as e:
            st.error(f"Failed to process tasks: {e}")
        finally:
            sys.stdout.flush()  # Ensure that the buffer is flushed at the end

    def setup_llm(self, llm_option, api_key):
        if llm_option == 'OpenAI GPT-4':
            return ChatOpenAI(model="gpt-4", api_key=api_key)
        elif llm_option == 'Claude-3':
            return ChatAnthropic(model="claude-3-haiku-20240307", api_key=api_key)
        else:  # Groq
            return ChatGroq(api_key=api_key, model_name="llama3-8b-8192")

    def initialize_agents_and_tasks(self, agent_details, tasks_list, llm):
        agents, tasks = [], []
        for detail, task_desc in zip(agent_details, tasks_list):
            role, goal, backstory = detail
            agent = Agent(role=role, goal=goal, backstory=backstory, verbose=True, allow_delegation=True, llm=llm, tools=[duckduckgo_search])
            task = Task(description=task_desc, expected_output="Report on " + task_desc, agent=agent)
            agents.append(agent)
            tasks.append(task)
        return agents, tasks

if __name__ == "__main__":
    app = CrewAIApp()
    app.run()
