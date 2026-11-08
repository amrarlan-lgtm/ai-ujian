# Q1b - Intelligent Agent (Model-based / Utility-based Reflex) pada Smart Home
class SmartHomeAgent:
    def __init__(self, target_temp=24):
        self.target_temp = target_temp
        self.state = {"temp": None, "occupied": False, "light": "off", "ac": "off"}  # model internal

    def perceive(self, percept):                 # SENSORS
        self.state.update(percept)

    def decide(self):                            # AGENT PROGRAM
        s, actions = self.state, []
        if s["occupied"]:
            if s["temp"] > self.target_temp + 1: actions.append(("ac", "on"))
            elif s["temp"] < self.target_temp - 1: actions.append(("ac", "off"))
            if s["lux"] < 100: actions.append(("light", "on"))
        else:                                    # hemat energi
            actions += [("ac", "off"), ("light", "off")]
        if s.get("smoke"): actions.append(("alarm", "on"))
        return actions

    def act(self, actions):                      # ACTUATORS
        for device, cmd in actions:
            self.state[device] = cmd
            print(f"  [ACTUATOR] {device} -> {cmd}")

agent = SmartHomeAgent()
scenarios = [
    {"temp": 30, "occupied": True,  "lux": 50,  "smoke": False},
    {"temp": 22, "occupied": True,  "lux": 300, "smoke": False},
    {"temp": 28, "occupied": False, "lux": 20,  "smoke": False},
    {"temp": 29, "occupied": True,  "lux": 200, "smoke": True},
]
for i, p in enumerate(scenarios, 1):
    print(f"Percept {i}: {p}")
    agent.perceive(p); agent.act(agent.decide())
