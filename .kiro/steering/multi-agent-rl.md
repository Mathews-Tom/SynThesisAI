---
inclusion: fileMatch
fileMatchPattern: '**/agent_*.py'
---

# Multi-Agent Reinforcement Learning Guidelines

## Multi-Agent RL Architecture

SynThesisAI implements a sophisticated multi-agent reinforcement learning (MARL) system with three specialized agents:

1. **Generator Agent**: Selects optimal content generation strategies
2. **Validator Agent**: Predicts content quality and provides feedback
3. **Curriculum Agent**: Ensures pedagogical coherence and learning progression

## Agent Implementation

### Base RL Agent

```python
class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _build_model(self):
        # Neural network for deep Q learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def update_policy(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Specialized Agents

#### Generator Agent

```python
class GeneratorRLAgent(RLAgent):
    def __init__(self):
        super().__init__(state_size=100, action_size=20)
        self.generation_strategies = self._initialize_strategies()
    
    def _initialize_strategies(self):
        # Define different content generation strategies
        return [
            {"approach": "step_by_step", "complexity": "low"},
            {"approach": "concept_based", "complexity": "medium"},
            {"approach": "problem_solving", "complexity": "high"},
            # Additional strategies...
        ]
    
    def select_generation_strategy(self, state):
        action_index = self.select_action(state)
        return self.generation_strategies[action_index]
```

#### Validator Agent

```python
class ValidatorRLAgent(RLAgent):
    def __init__(self):
        super().__init__(state_size=150, action_size=10)
        self.validation_thresholds = np.linspace(0.5, 0.95, 10)
    
    def predict_quality(self, content):
        # Predict content quality and provide feedback
        state = self.extract_features(content)
        action_index = self.select_action(state)
        threshold = self.validation_thresholds[action_index]
        
        # Assess content against threshold
        quality_assessment = self.assess_quality(content)
        feedback = self.generate_feedback(content, quality_assessment, threshold)
        
        return {
            "quality_score": quality_assessment["overall_score"],
            "passes_threshold": quality_assessment["overall_score"] >= threshold,
            "feedback": feedback
        }
```

#### Curriculum Agent

```python
class CurriculumRLAgent(RLAgent):
    def __init__(self):
        super().__init__(state_size=200, action_size=15)
        self.curriculum_strategies = self._initialize_strategies()
    
    def _initialize_strategies(self):
        # Define different curriculum strategies
        return [
            {"progression": "linear", "difficulty_curve": "gradual"},
            {"progression": "spiral", "difficulty_curve": "oscillating"},
            {"progression": "mastery", "difficulty_curve": "threshold_based"},
            # Additional strategies...
        ]
    
    def suggest_improvements(self, request):
        state = self.extract_features(request)
        action_index = self.select_action(state)
        strategy = self.curriculum_strategies[action_index]
        
        return self.generate_curriculum_guidance(request, strategy)
```

## Coordination Mechanism

```python
class MultiAgentRLCoordinator:
    def __init__(self):
        self.generator_agent = GeneratorRLAgent()
        self.validator_agent = ValidatorRLAgent()
        self.curriculum_agent = CurriculumRLAgent()
        self.coordination_policy = CoordinationPolicy()
    
    async def coordinate_generation(self, generator, request):
        # Multi-agent state observation
        state = self.observe_environment(request)
        
        # Agent action selection
        actions = {
            'generator': self.generator_agent.select_action(state),
            'validator': self.validator_agent.select_action(state),
            'curriculum': self.curriculum_agent.select_action(state)
        }
        
        # Coordination mechanism
        coordinated_action = self.coordination_policy.coordinate(actions, state)
        
        # Execute coordinated action
        content = await generator.generate(coordinated_action)
        
        # Multi-agent learning update
        rewards = self.calculate_rewards(content, request.quality_requirements)
        for agent_name, agent in [
            ('generator', self.generator_agent),
            ('validator', self.validator_agent),
            ('curriculum', self.curriculum_agent)
        ]:
            agent.update_policy(
                state, 
                actions[agent_name], 
                rewards[agent_name], 
                self.observe_environment(content), 
                False
            )
        
        return content
```

## Best Practices for MARL Implementation

1. **State Representation**: Design comprehensive state representations that capture relevant features
2. **Reward Function Design**: Create multi-objective reward functions aligned with quality metrics
3. **Exploration-Exploitation Balance**: Implement proper epsilon decay for balanced exploration
4. **Coordination Mechanisms**: Design effective coordination protocols to avoid conflicts
5. **Experience Replay**: Use replay buffers to improve learning stability
6. **Hyperparameter Tuning**: Optimize learning rates, discount factors, and network architectures
