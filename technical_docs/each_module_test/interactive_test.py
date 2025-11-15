# interactive_test.py
# Interactive Testing Console
# Interactive Test Shell

import asyncio
from main import CognitiveTextBrain
import cmd


class BrainTestShell(cmd.Cmd):
    intro = """
    🧠 Cognitive Brain Interactive Test Console 🧠

    Commands:
    - test <module>  : Test a specific module
    - query <text>   : Send a query to the brain
    - state <region> : Show state of a brain region
    - metrics        : Show current metrics
    - help           : Show available commands
    - exit           : Exit the console
    """
    prompt = '(brain-test) '

    def __init__(self):
        super().__init__()
        self.brain = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def do_init(self, arg):
        """Initialize the brain: init"""
        print("Initializing brain...")
        self.brain = CognitiveTextBrain()
        self.loop.run_until_complete(self.brain.initialize())
        print("Brain initialized!")

    def do_test(self, module):
        """Test a specific module: test working_memory"""
        if not self.brain:
            print("Please initialize the brain first with 'init'")
            return

        test_functions = {
            "working_memory": self._test_working_memory,
            "reasoning": self._test_reasoning,
            "language": self._test_language,
            "executive": self._test_executive
        }

        if module in test_functions:
            self.loop.run_until_complete(test_functions[module]())
        else:
            print(f"Unknown module: {module}")
            print(f"Available modules: {', '.join(test_functions.keys())}")

    def do_query(self, text):
        """Send a query to the brain: query What is consciousness?"""
        if not self.brain:
            print("Please initialize the brain first with 'init'")
            return

        print(f"\nProcessing: {text}")
        response = self.loop.run_until_complete(self.brain.process_text(text))
        print(f"\nResponse: {response}")

    def do_state(self, region):
        """Show state of a brain region: state working_memory"""
        if not self.brain:
            print("Please initialize the brain first with 'init'")
            return

        if region in self.brain.regions:
            state = self.brain.regions[region].get_state()
            print(f"\n{region.upper()} State:")
            for key, value in state.items():
                print(f"  {key}: {value}")
        else:
            print(f"Unknown region: {region}")
            print(f"Available regions: {', '.join(self.brain.regions.keys())}")

    def do_metrics(self, arg):
        """Show current metrics: metrics"""
        print("\nCurrent Metrics:")
        print("  [Metrics would be displayed here from Prometheus]")
        print("  Working Memory: 71%")
        print("  Reasoning Confidence: 0.85")
        print("  Active Goals: 3")

    def do_exit(self, arg):
        """Exit the console: exit"""
        if self.brain:
            print("Shutting down brain...")
            self.loop.run_until_complete(self.brain.shutdown())
        print("Goodbye!")
        return True

    async def _test_working_memory(self):
        """Test working memory module"""
        wm = self.brain.regions['working_memory']

        print("\nTesting Working Memory:")

        # Add items
        items = [
            {"content": {"text": "Test item 1"}},
            {"content": {"text": "Test item 2"}},
            {"content": {"text": "Test item 3"}}
        ]

        for item in items:
            await wm.store(item)
            print(f"Stored: {item['content']['text']}")

        # Check state
        state = wm.get_state()
        print(f"\nBuffer size: {state['buffer_size']}")
        print(f"Capacity usage: {state['capacity_usage']:.1%}")

    async def _test_reasoning(self):
        """Test reasoning modules"""
        print("\nTesting Reasoning:")

        problem = "If A is true and A implies B, is B true?"

        # Test logical reasoning
        logic = self.brain.regions['logical_reasoning']
        result = await logic.reason(problem, {})

        print(f"Problem: {problem}")
        print(f"Conclusion: {result.get('conclusion', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")

    async def _test_language(self):
        """Test language processing"""
        print("\nTesting Language Processing:")

        text = "The cognitive brain processes information through multiple pathways."

        lang = self.brain.regions['language']
        result = await lang.process({"text": text})

        print(f"Text: {text}")
        print(f"Tokens: {len(result['tokens'])}")
        print(f"Entities: {result['entities']}")
        print(f"Complexity: {result['complexity_score']:.2f}")

    async def _test_executive(self):
        """Test executive functions"""
        print("\nTesting Executive Functions:")

        executive = self.brain.regions['prefrontal_cortex']
        state = executive.get_state()

        print(f"Active goals: {state['active_goals']}")
        print(f"Current plan: {state['current_plan']}")
        print(f"Success rate: {state['recent_success_rate']:.1%}")


if __name__ == '__main__':
    BrainTestShell().cmdloop()