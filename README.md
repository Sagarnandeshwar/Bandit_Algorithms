# Bandit Algorithms
Implemented Epsilon Greedy, UCB (Upper Confidence Bound), and Thompson sampling to understand and experiment Bandit algorithms in Bernoulli bandit with k arms simulation. 

## Bandit Algorithms 
Bandit algorithms are a class of sequential decision-making algorithms used to solve the multi-armed bandit problem. In this problem, an agent (or decision-maker) is faced with a set of k arms (actions) with unknown reward distributions. The agent's goal is to learn which arms yield the highest rewards and maximize the total reward accumulated over time. 

![1](https://github.com/Sagarnandeshwar/Bandit_Algorithms/blob/main/image/1.png)

The choice of the bandit algorithm depends on the problem context, the level of uncertainty about the arms' reward probabilities, and the trade-off between exploration and exploitation desired by the decision-maker. Some algorithms perform better in specific scenarios, so it's essential to consider the characteristics of the bandit problem when selecting an appropriate algorithm. 

## Simulation: Bernoulli bandit with k arms 
In Bernoulli bandit problem with k arms, we are faced with k different actions (arms) to choose from, and each arm generates a binary reward (0 or 1) with an unknown probability of success. The goal is to maximize the total reward obtained by choosing the best arms while learning about their success probabilities through exploration. 

As stated before, the Bernoulli bandit problem can be represented as follows: 
- There are k arms, numbered from 1 to k. 
- Each arm i has an unknown probability p_i of yielding a reward of 1 (success) when pulled. 
- The rewards obtained from each arm are independent and identically distributed (i.i.d.) Bernoulli random variables with parameter p_i. 

The objective is to find the best arm(s) with the highest probability of success, given the limited information gathered from pulling arms and observing their outcomes. 

Solving the Bernoulli bandit problem involves choosing appropriate strategies or algorithms to balance exploration (trying out different arms to gather information) and exploitation (selecting arms that have performed well so far to maximize immediate rewards) 

## Hyperparameters and choices  
An update function is used to update the estimated reward probabilities for each arm based on the observed rewards. The update function is crucial in learning from the outcomes of arm pulls and improving the agent's knowledge about the arms' success probabilities over time. 

Let's consider a simple update function that calculates the average reward for each arm. This update function is commonly used in algorithms like the epsilon-greedy and UCB. 

Assuming that we have an array rewards[i] to store the rewards obtained from arm i and an array pulls[i] to store the number of times arm i has been pulled, the update function for arm i can be defined as follows: 

When arm i is pulled, and a reward reward_i (0 or 1) is observed: 
- Increment the number of times arm i has been pulled: pulls[i] += 1 
- Update the sum of rewards obtained from arm i: rewards[i] += reward_i
- 
The estimated probability of success for arm i, denoted as p_hat_i, can be calculated as the average of rewards obtained from arm i divided by the total number of times it has been pulled: p_hat_i = rewards[i] / pulls[i] 

The p_hat_i represents the agent's current estimate of the success probability for arm i based on the available data. As the agent explores and exploits different arms over time, the update function continuously improves the estimates of p_hat_i, allowing the agent to make more informed decisions. 

## epsilon-Greedy Algorithm

The epsilon-greedy algorithm is a common strategy used to solve the exploration-exploitation trade-off in the multi-armed bandit problem. The algorithm allows an agent to explore different actions (arms) with some probability epsilon (ε) while exploiting the currently estimated best action(s) most of the time. 

Here's a step-by-step explanation of the epsilon-greedy algorithm: 

1. **Initialization:** Set the exploration parameter epsilon (ε) to a value between 0 and 1. This parameter determines the proportion of exploration versus exploitation. Typically, a small value of epsilon (e.g., 0.1) is used to ensure that exploitation dominates once the algorithm has gathered sufficient information. 

2. **Arm Selection:**
   - With a probability of ε, choose a random arm uniformly at random from all available arms. This is the exploration step, where the agent explores new arms to gather information about their reward probabilities.
   - With a probability of (1 - ε), choose the arm with the highest estimated probability of success (i.e., the arm that has yielded the highest average rewards so far). This is the exploitation step, where the agent selects the best-known arm to maximize immediate rewards. 

3. **Pull the chosen arm and observe the reward.**

4. **Update the estimates:**
   - Keep track of the rewards obtained from each arm.
   - Update the estimated probability of success for the chosen arm based on the observed reward. This is typically done using some form of average or weighted average of the observed rewards for that arm. 

5. **Repeat steps 2 to 4 for a certain number of iterations or time steps.**

![2](https://github.com/Sagarnandeshwar/Bandit_Algorithms/blob/main/image/2.png)
![3](https://github.com/Sagarnandeshwar/Bandit_Algorithms/blob/main/image/3.png)

The epsilon-greedy algorithm balances exploration and exploitation over time. As the algorithm gains more information about the arms through exploration, it gradually shifts towards exploiting the best-performing arms to maximize long-term rewards. 

## UCB algorithm 

The UCB (Upper Confidence Bound) algorithm is another popular approach for solving the multi-armed bandit problem. It addresses the exploration-exploitation trade-off by using uncertainty estimates to guide the arm selection process. The algorithm aims to efficiently balance exploration of uncertain arms and exploitation of arms that seem promising based on available data. 

Here's how the UCB algorithm works: 
1. **Initialization:** Initialize the number of times each arm has been pulled to zero, and the estimated mean reward for each arm to zero. 
2. **Arm Selection:**
   - For each arm i, calculate the upper confidence bound (UCB) value based on the observed rewards and the number of times arm i has been pulled.
   - The UCB value for arm i is calculated as: UCB(i) = mean_reward(i) + sqrt(2 * log(total_pulls) / arm_pulls(i)), where mean_reward(i) is the average reward obtained from arm i, total_pulls is the total number of arms pulled so far, and arm_pulls(i) is the number of times arm i has been pulled.
   - Choose the arm with the highest UCB value for the next action.
3. **Pull the chosen arm and observe the reward.**
4. **Update the estimates:**
   - Update the number of times the chosen arm has been pulled.
   - Update the estimated mean reward for the chosen arm based on the observed reward. 
5. **Repeat steps 2 to 4 for a certain number of iterations or time steps.**

The UCB algorithm balances exploration and exploitation by selecting arms with a high UCB value. The UCB value is based on the estimated mean reward and the confidence interval, represented by the term sqrt(2 * log(total_pulls) / arm_pulls(i)), which increases as an arm is pulled less frequently. The algorithm encourages exploration of arms with high uncertainty (uncertain mean rewards) while favoring arms with potentially high rewards based on the observed data. 

![4](https://github.com/Sagarnandeshwar/Bandit_Algorithms/blob/main/image/4.png)

Over time, as the algorithm explores different arms and gathers more information, it becomes increasingly confident in its estimates, and the exploitation of promising arms dominates over exploration. 

## Thompson sampling 
Thompson Sampling is another popular algorithm for solving the multi-armed bandit problem. Unlike the epsilon-greedy and UCB algorithms, Thompson Sampling uses a probabilistic approach to balance exploration and exploitation. It makes decisions based on posterior probability distributions, which represent the agent's uncertainty about the true reward probabilities of each arm. 

Here's how the Thompson Sampling algorithm works: 
1. **Initialization:** Assume a prior distribution for each arm's reward probability. This distribution should capture the agent's uncertainty about the arm's performance before any observations are made. A common choice is the Beta distribution, which is well-suited for modeling probabilities. 
2. **Arm Selection:**
   - For each arm i, sample a random value from its corresponding posterior distribution (e.g., a random value from the Beta distribution for arm i). This sampled value represents a potential reward probability for that arm.
   - Choose the arm with the highest sampled value for the next action. 
3. **Pull the chosen arm and observe the reward.**
4. **Update the posterior distribution:**
   - Based on the observed reward, update the posterior distribution for the chosen arm using Bayesian updating. For example, if the prior distribution is Beta(alpha, beta) and the observed reward is 1 (success) or 0 (failure), the updated posterior distribution becomes Beta(alpha + 1, beta) or Beta(alpha, beta + 1), respectively. 
5. **Repeat steps 2 to 4 for a certain number of iterations or time steps.**

![5](https://github.com/Sagarnandeshwar/Bandit_Algorithms/blob/main/image/5.png)

The Thompson Sampling algorithm works by maintaining a distribution over the reward probabilities for each arm. The sampling step (Step 2) allows it to explore arms in proportion to their potential performance, as arms with higher expected rewards have a higher chance of being selected. As the algorithm collects more data and updates the posterior distributions, it becomes more confident in its estimates and shifts towards exploiting the arms with higher expected rewards. 


