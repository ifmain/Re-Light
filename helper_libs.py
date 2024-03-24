class Logger:
    def __init__(self):
        self.rules = {
            'main': True,
            'libs': True
        }
        
    def create_logger_for(self, rule_name):
        def log_func(msg):
            if self.rules.get(rule_name, False):
                print(msg)
        return log_func

    class Rule:
        def __init__(self, parent_logger):
            self.parent_logger = parent_logger

        def main(self, state):
            self.parent_logger.rules['main'] = state

        def libs(self, state):
            self.parent_logger.rules['libs'] = state

    def __getattr__(self, item):
        if item == 'rule':
            return Logger.Rule(self)
        raise AttributeError(f"'Logger' object has no attribute '{item}'")

logger = Logger()