# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from typing import List

import util
from game import Directions


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # ITERATIVE VERSION (UNCOMMENT TO USE)
    # Initialize the fringe (stack) with the start state
    # Each element in the stack is a tuple: (state, list_of_actions)
    fringe = util.Stack()
    fringe.push((problem.getStartState(), []))

    # Keep track of visited states to avoid cycles
    visited = set()

    while not fringe.isEmpty():
        # Pop the deepest node from the stack
        current_state, actions = fringe.pop()

        # Skip if already visited
        if current_state in visited:
            continue

        # Mark current state as visited
        visited.add(current_state)

        # Check if we reached the goal
        if problem.isGoalState(current_state):
            return actions

        # Expand the current node and push successors onto the stack
        # Push in the order provided by getSuccessors (important for correct path length)
        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited:
                # Create new action list by appending current action
                new_actions = actions + [action]
                fringe.push((successor, new_actions))

    # If no solution found, return empty list
    return []

    # RECURSIVE VERSION (COMMENTED OUT - UNCOMMENT TO USE)
    """
    def dfs_recursive(state, visited, actions):
        # Mark current state as visited
        visited.add(state)

        # Check if we reached the goal
        if problem.isGoalState(state):
            return actions

        # Explore successors
        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                # Recursively search from successor
                result = dfs_recursive(successor, visited, actions + [action])
                if result is not None:
                    return result

        # No solution found from this path
        return None

    # Initialize visited set and start recursive search
    visited = set()
    result = dfs_recursive(problem.getStartState(), visited, [])

    # Return result or empty list if no solution
    return result if result is not None else []
    """


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    # Initialize the fringe (queue) with the start state
    # Each element in the queue is a tuple: (state, list_of_actions)
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))

    # Keep track of visited states to avoid cycles
    visited = set()

    while not fringe.isEmpty():
        # Dequeue the shallowest node from the queue
        current_state, actions = fringe.pop()

        # Skip if already visited
        if current_state in visited:
            continue

        # Mark current state as visited
        visited.add(current_state)

        # Check if we reached the goal
        if problem.isGoalState(current_state):
            return actions

        # Expand the current node and enqueue successors
        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited:
                # Create new action list by appending current action
                new_actions = actions + [action]
                fringe.push((successor, new_actions))

    # If no solution found, return empty list
    return []


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
#     """Search the node of least total cost first."""

#     # Initialize the fringe (priority queue) with the start state
#     # Each element is a tuple: (state, list_of_actions, total_cost)
#     # Priority queue orders by total_cost
#     fringe = util.PriorityQueue()
#     fringe.push((problem.getStartState(), [], 0), 0)

#     # Keep track of visited states to avoid cycles
#     visited = set()

#     while not fringe.isEmpty():
#         # Pop the node with the lowest total cost
#         current_state, actions, current_cost = fringe.pop()

#         # Skip if already visited
#         if current_state in visited:
#             continue

#         # Mark current state as visited
#         visited.add(current_state)

#         # Check if we reached the goal
#         if problem.isGoalState(current_state):
#             return actions

#         # Expand the current node and add successors to priority queue
#         for successor, action, stepCost in problem.getSuccessors(current_state):
#             if successor not in visited:
#                 # Calculate new total cost
#                 new_cost = current_cost + stepCost
#                 # Create new action list by appending current action
#                 new_actions = actions + [action]
#                 # Push with priority equal to total cost
#                 fringe.push((successor, new_actions, new_cost), new_cost)

#     # If no solution found, return empty list
#     return []


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
#     """Search the node that has the lowest combined cost and heuristic first."""

#     # Initialize the fringe (priority queue) with the start state
#     # Each element is a tuple: (state, list_of_actions, total_cost)
#     # Priority queue orders by f(n) = g(n) + h(n)
#     fringe = util.PriorityQueue()
#     start_state = problem.getStartState()
#     fringe.push((start_state, [], 0), 0 + heuristic(start_state, problem))

#     # Keep track of visited states to avoid cycles
#     visited = set()

#     while not fringe.isEmpty():
#         # Pop the node with the lowest f(n) = g(n) + h(n)
#         current_state, actions, current_cost = fringe.pop()

#         # Skip if already visited
#         if current_state in visited:
#             continue

#         # Mark current state as visited
#         visited.add(current_state)

#         # Check if we reached the goal
#         if problem.isGoalState(current_state):
#             return actions

#         # Expand the current node and add successors to priority queue
#         for successor, action, stepCost in problem.getSuccessors(current_state):
#             if successor not in visited:
#                 # Calculate new total cost g(n)
#                 new_cost = current_cost + stepCost
#                 # Create new action list by appending current action
#                 new_actions = actions + [action]
#                 # Calculate f(n) = g(n) + h(n)
#                 # g(n) = new_cost (actual cost from start to successor)
#                 # h(n) = heuristic(successor, problem) (estimated cost from successor to goal)
#                 priority = new_cost + heuristic(successor, problem)
#                 # Push with priority equal to f(n)
#                 fringe.push((successor, new_actions, new_cost), priority)

#     # If no solution found, return empty list
#     return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
