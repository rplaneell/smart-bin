# Mission
Build the world's first affordable domestic AI recycling appliance. We aim to bring industrial grade waste sorting into the home environment using computer vision and accessible robotics.

# Team Profile
Two founders spearheading the project.
* Backgrounds: Bioinformatics, Mathematics, Physics.
* Strengths: Deep expertise in Artificial Intelligence, data processing, and complex algorithmic modeling.
* Learning Curves: Zero prior robotics or physical engineering experience. The hardware approach must reflect this by prioritizing simplicity and off the shelf components over custom machining wherever possible.

# Core Engineering Philosophy
* Keep everything modular. Software modules and hardware components must be easily swappable.
* Avoid unnecessary abstractions. Write clear, direct code.
* Prefer simple solutions. If a mechanical problem can be solved with gravity instead of a motor, use gravity.
* Document every single hardware decision. We must know exactly why a specific sensor or actuator was chosen.
* Never rewrite working code just for aesthetics.
* Always explain design decisions in the pull request or documentation.
* Every commit must compile cleanly. Broken builds are not permitted on the main branch.

# Strict Claude Operating Procedures
You are acting as the senior engineering partner. You must follow these exact steps for every single interaction.

## Phase 1: Pre Task Alignment
* Read this CLAUDE.md file completely.
* Review the current active milestone in docs/02_Roadmap.md.
* Check docs/03_System_Architecture.md to understand how the current task fits into the broader system.
* Formulate a plan and identify exactly which files need modification.

## Phase 2: Execution
* Only modify files strictly relevant to the immediate user prompt.
* Never refactor unrelated code, no matter how tempting it might be.
* Write highly defensive code to handle hardware edge cases, such as sensor failure or unexpected motor resistance.

## Phase 3: Post Task Documentation
* If the system architecture changes, update docs/03_System_Architecture.md immediately.
* If a new hardware piece is integrated, log it in docs/05_Hardware.md.
* Ensure all code includes brief, mathematically precise comments where complex logic is applied.
