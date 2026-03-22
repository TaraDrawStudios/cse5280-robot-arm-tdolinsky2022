# Robot Crowd Interaction Report

## Clustering Approach
The clustering approach used in this simulation is based on identifying agents (people) who are near the exits of the building. Instead of using a complex clustering algorithm like K-means, the system filters agents based on their position and selects only those within defined exit zones. Once these agents are identified, the system computes the cluster center by averaging the positions of all agents near the exit. This provides a simple and efficient way to determine where the crowd is forming.

This approach works well because the main area of ​​interest is near the exits, where crowd congestion is most likely to occur. By focusing only on agents near exits, the robot can respond to important crowd behavior instead of tracking every individual in the building.

## Prediction Model
The prediction model estimates where the crowd will move in the near future. This is done by calculating the average velocity of the clustered agents and projecting their future position using a simple motion model:

**Predicted Position = Average Position + (Average Velocity × Time Step)**

This allows the robot to move toward where the crowd is going, not just where it currently is. This predictive behavior helps the robot react earlier and position itself more effectively.

## How the Robot Interacts with the Crowd
The robot interacts with the crowd by moving its end-effector toward the predicted crowd position near the exits. The robot does not physically push agents but instead acts as a dynamic obstacle. Agents detect the robot's position and receive a movement penalty when they get too close, which causes them to move around the robot.

This interaction helps guide crowd movement and prevents congestion directly at the exit by slightly redirecting people away from crowded areas.

## Observations from Experiments
From the experiments, several behaviors were observed:

- Agents naturally formed clusters near exits.
- When the robot moved toward the predicted crowd position, it helped reduce congestion near exits.
- The prediction model worked better than only moving toward the current crowd center because it allowed the robot to arrive earlier.
- If the robot moved too slowly, it had little effect on the crowd.
- If the robot was positioned well near exits, agents spread out more and exited more smoothly.
- The crowd avoidance behavior between agents also helped prevent agents from overlapping and created more realistic movement.

Overall, the robot successfully influenced crowd flow by predicting crowd movement and positioning itself strategically near exits.


Linke: https://drive.google.com/file/d/1WcDFibJwfVidsgsyfkynkZyQfLHwT_Ul/view?usp=sharing
