import streamlit as st
import sys
import re
import os

from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchRun

from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool,
    SeleniumScrapingTool,
)

# StreamToExpander class to capture stdout and display in Streamlit expander
class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit

    def write(self, data):
        # Clean ANSI escape codes from output
        cleaned_data = re.sub(r'\x1B\[\d+;?\d*m', '', data)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.pop(0)
        self.buffer.append(cleaned_data)

        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

    def flush(self):
        if self.buffer:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

# Main application class
class CrewAIApp:
    def __init__(self):
        # LLM and tool options
        self.llm_options = ['OpenAI GPT-4o-mini', 'OpenAI GPT-4o', 'Claude', 'Groq']
        self.tool_options = [
            'DuckDuckGoSearchRun',
            'SerperDevTool',
            'WebsiteSearchTool',
            'ScrapeWebsiteTool',
            'SeleniumScrapingTool',
        ]

    def run(self):
        st.set_page_config(page_title="CrewAI Agents Builder", layout="wide")
        st.title("CrewAI Agents Builder")

        # Sidebar configuration
        st.sidebar.header("Configuration")

        # LLM configuration
        llm_option = st.sidebar.selectbox("Choose LLM for Agents:", self.llm_options)
        api_key = st.sidebar.text_input("Enter API Key for chosen LLM:", type="password")

        # Agent and Task configuration
        number_of_agents = st.sidebar.number_input(
            "Number of Agents to Create:", min_value=1, max_value=10, value=1, step=1
        )

        # Process selection
        process = st.sidebar.selectbox('Assign Process to a Crew', ['Sequential Process', 'Hierarchical Process'])

        # Tools selection
        tools_list = st.sidebar.multiselect("Select tools you want to use", self.tool_options)
        all_tools_list = self.manage_tool(tools_list) if tools_list else []

        # Main content
        agent_details = self.collect_agent_details(number_of_agents)
        task_details = self.collect_task_details(number_of_agents)

        # PDF Analysis
        number_of_pdfs = st.number_input(
            "Number of PDFs for Analysis:", min_value=0, max_value=10, value=0, step=1
        )
        pdfs = self.upload_pdfs(number_of_pdfs)

        # Execution
        if st.button("Start Crew Execution"):
            if not api_key:
                st.error("API Key is required.")
            elif not agent_details or not task_details:
                st.error("All agent details and tasks are required.")
            else:
                self.run_crew_analysis(
                    agent_details, task_details, llm_option, api_key, pdfs, all_tools_list, process
                )

    def collect_agent_details(self, number_of_agents):
        agent_details = []
        for i in range(number_of_agents):
            with st.expander(f"Agent {i+1} Details", expanded=True):
                role = st.text_input(f"Role for Agent {i+1}", key=f"role_{i}")
                goal = st.text_area(f"Goal for Agent {i+1}", key=f"goal_{i}")
                backstory = st.text_area(f"Backstory for Agent {i+1}", key=f"backstory_{i}")
                agent_details.append((role, goal, backstory))
        return agent_details

    def collect_task_details(self, number_of_agents):
        task_details = []
        for i in range(number_of_agents):
            with st.expander(f"Task {i+1} Details", expanded=True):
                description = st.text_input(f"Description for Task {i+1}", key=f"description_{i}")
                expected_output = st.text_area(f"Expected output for Task {i+1}", key=f"expected_output_{i}")
                task_details.append((description, expected_output))
        return task_details

    def upload_pdfs(self, number_of_pdfs):
        pdfs = []
        for i in range(number_of_pdfs):
            pdf = st.file_uploader(f"Upload PDF {i+1} for Analysis", type=['pdf'], key=f'pdf_{i}')
            if pdf:
                pdfs.append(pdf)
        return pdfs

    def run_crew_analysis(self, agent_details, task_details, llm_option, api_key, pdfs, all_tools_list, process):
        process_output_expander = st.expander("Processing Output:")
        sys.stdout = StreamToExpander(process_output_expander)

        try:
            llm = self.setup_llm(llm_option, api_key)
            agents, tasks = self.initialize_agents_and_tasks(
                agent_details, task_details, llm, all_tools_list
            )

            if not agents or not tasks:
                st.error("Failed to initialize agents or tasks.")
                return

            # Append PDF contents to the first agent's task description
            if pdfs and tasks:
                for pdf in pdfs:
                    try:
                        import PyPDF2  # Moved import inside the function
                        pdf_reader = PyPDF2.PdfReader(pdf)
                        pdf_text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            pdf_text += page.extract_text()
                        tasks[0].description += "\n\n" + pdf_text
                    except Exception as e:
                        st.error(f"Failed to read PDF contents: {e}")

            if process == 'Sequential Process':
                crew = Crew(agents=agents, tasks=tasks, verbose=True, process=Process.sequential)
            else:
                crew = Crew(
                    agents=agents, tasks=tasks, verbose=True, process=Process.hierarchical, manager_llm=llm
                )
            crew_result = crew.kickoff()

            st.write(crew_result)
        except Exception as e:
            st.error(f"Failed to process tasks: {e}")
        finally:
            sys.stdout.flush()

    def setup_llm(self, llm_option, api_key):
        if llm_option == 'OpenAI GPT-4o-mini':
            return ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
        elif llm_option == 'OpenAI GPT-4o':
            return ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key)
        elif llm_option == 'Claude':
            # Placeholder for Claude LLM implementation
            st.error("Claude LLM integration is not implemented yet.")
            return None
        elif llm_option == 'Groq':
            # Placeholder for Groq LLM implementation
            st.error("Groq LLM integration is not implemented yet.")
            return None
        else:
            raise ValueError("Unsupported LLM option selected.")

    def initialize_agents_and_tasks(self, agent_details, task_details, llm, all_tools_list):
        agents = []
        tasks = []
        for detail, task_desc in zip(agent_details, task_details):
            role, goal, backstory = detail
            description, expected_output = task_desc

            if not role or not goal:
                st.error("Role and Goal are required for each agent.")
                continue

            agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=True,
                allow_delegation=True,
                llm=llm,
                tools=all_tools_list
            )
            task = Task(
                description=description,
                expected_output=expected_output,
                agent=agent
            )
            agents.append(agent)
            tasks.append(task)
        return agents, tasks

    def manage_tool(self, selected_tools):
        tool_list = []
        for tool_name in selected_tools:
            if tool_name == 'DuckDuckGoSearchRun':
                duckduckgo_search = DuckDuckGoSearchRun()
                tool_list.append(duckduckgo_search)
            elif tool_name == 'SerperDevTool':
                serper_api_key = st.sidebar.text_input("Enter Serper API Key", type="password")
                if serper_api_key:
                    os.environ['SERPER_API_KEY'] = serper_api_key
                    search_tool = SerperDevTool()
                    tool_list.append(search_tool)
                else:
                    st.error("Serper API Key is required for SerperDevTool.")
            elif tool_name == 'WebsiteSearchTool':
                web_search_tool = WebsiteSearchTool()
                tool_list.append(web_search_tool)
            elif tool_name == 'ScrapeWebsiteTool':
                scrape_website_tool = ScrapeWebsiteTool()
                tool_list.append(scrape_website_tool)
            elif tool_name == 'SeleniumScrapingTool':
                selenium_tool = SeleniumScrapingTool()
                tool_list.append(selenium_tool)
        return tool_list

if __name__ == "__main__":
    app = CrewAIApp()
    app.run()
