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

    ollama_llm = LLM(
        model="ollama/deepseek-r1",
        base_url="http://localhost:11434",
        temperature=0.7,
    )

    @agent
    def synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["synthesizer"], 
            verbose=True,
            llm=self.ollama_llm,
        )

    @agent
    def newsletter_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["newsletter_writer"],
            tools=[WordCounterTool()],
            verbose=True,
            llm=self.ollama_llm
        )

    @agent
    def newsletter_editor(self) -> Agent:
        return Agent(
            config=self.agents_config["newsletter_editor"],
            tools=[WordCounterTool()],
            verbose=True,
            llm=self.ollama_llm
        )

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
            # chat_llm="gpt-4o",
            chat_llm=self.ollama_llm,

        )
