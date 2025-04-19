from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv
load_dotenv()

from convo_newsletter_crew.tools.word_counter_tool import WordCounterTool


@CrewBase
class ConvoNewsletterCrew:
    """ConvoNewsletterCrew crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        """Initialize the crew with configured LLM"""
        self.llm = self.get_llm()

    def get_llm(self):
        """Get the appropriate LLM based on environment configuration"""
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        
        try:
            if llm_provider == "ollama":
                return LLM(
                    model=os.getenv("OLLAMA_MODEL", "ollama/mistral"),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                    request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "300")),
                    max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
                )
            elif llm_provider == "openai":
                return LLM(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            # Fallback to a default LLM or raise the exception
            raise

    def create_agent(self, agent_type, tools=None, **kwargs):
        """Factory method to create agents with consistent configuration"""
        return Agent(
            config=self.agents_config[agent_type],
            verbose=kwargs.get("verbose", True),
            llm=kwargs.get("llm", self.llm),
            tools=tools or [],
        )

    @agent
    def synthesizer(self) -> Agent:
        return self.create_agent("synthesizer")

    @agent
    def newsletter_writer(self) -> Agent:
        return self.create_agent("newsletter_writer", tools=[WordCounterTool()])

    @agent
    def newsletter_editor(self) -> Agent:
        return self.create_agent("newsletter_editor", tools=[WordCounterTool()])
    
    # If you want the agent to have different LLM provider that rest of the agents.
    # @agent
    # def newsletter_editor(self) -> Agent:
    #     # Create a specific LLM for this agent only
    #     try:
    #         editor_llm = LLM(
    #             model=os.getenv("EDITOR_MODEL", "gpt-4o"),
    #             api_key=os.getenv("OPENAI_API_KEY"),
    #             temperature=float(os.getenv("EDITOR_TEMPERATURE", "0.5")),
    #         )
    #     except Exception as e:
    #         print(f"Error initializing editor LLM: {e}")
    #         # Fall back to the default LLM
    #         editor_llm = self.llm
        
    #     return self.create_agent("newsletter_editor", 
    #                             tools=[WordCounterTool()], 
    #                             llm=editor_llm)




    @task
    def generate_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline_task"],
        )

    @task
    def write_newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_newsletter_task"],
            output_file="newsletter_draft.md",
        )

    @task
    def review_newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_newsletter_task"],
            output_file="final_newsletter.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ConvoNewsletterCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm="gpt-4o-mini",
        )