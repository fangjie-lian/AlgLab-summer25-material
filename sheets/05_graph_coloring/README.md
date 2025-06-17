# Benchmarking the Performance of Different Solvers and Formulations for the Graph Coloring Problem

For the remainder of the semester, we will focus on the **graph coloring problem**, which arises as a subproblem in various applications, such as register allocation in compilers, scheduling, and frequency assignment in wireless networks.

Numerous formulations have been proposed for solving the graph coloring problem using different solvers. Nevertheless, the problem remains computationally challenging, making it a suitable candidate for comparing the performance of various solvers and formulations—an endeavor that, as you will see, is far from trivial.

In contrast to sorting algorithms, where performance comparisons are relatively straightforward—for example, by measuring the average time to sort *n* elements for increasing values of *n*—graph coloring presents a more complex landscape. Some small graphs may remain unsolved within a reasonable time limit, while some large graphs can be solved within seconds. Moreover, solver performance may vary significantly across different graph classes: a solver might perform well on one class but poorly on another.

## The Graph Coloring Problem

Given an undirected graph $G = (V, E)$ , the goal is to assign colors to the vertices such that:

- No two adjacent vertices share the same color.
- The total number of colors used is minimized.

Formally, we seek a function $c : V \rightarrow \{1, \dots, k\}$  such that $c(u) \neq c(v)$  for all edges $(u, v) \in E$ , and $k$  is minimized. The minimum such $k$  is called the *chromatic number* of the graph, denoted $\chi(G)$ .

This problem arises in a variety of contexts, including register allocation, scheduling, and frequency assignment in wireless networks.

### Why Is Graph Coloring Interesting?

The graph coloring problem is NP-hard. Even for small graphs, determining the chromatic number can be computationally demanding. Moreover, it is hard to approximate: unless $\text{P} = \text{NP}$ , there exists no polynomial-time algorithm that guarantees a good approximation factor in the general case.

This makes the problem an excellent benchmark for evaluating the performance of exact algorithms, such as:

- Integer Linear Programming (ILP) formulations
- Constraint Programming (CP) models
- SAT-based encodings

Each of these approaches has distinct strengths depending on the structure of the input graph.

### Which Metrics Should Be Considered for Benchmarking?

When benchmarking graph coloring algorithms, consider the following metrics:

- **Time to Proven Optimality**: How long does the solver take to find and prove the optimal solution?
- **Best Solution Within Time Limit**: What is the best solution found within a given time limit?
- **Best Bound Within Time Limit**: What is the best lower bound on the chromatic number found within the time limit?

## Tasks

1. Read the paper [*New Integer Linear Programming Models for the Vertex Coloring Problem*](https://arxiv.org/pdf/1706.10191) to familiarize yourself with ILP formulations for graph coloring.
2. Read the chapter [*Benchmarking Your Model*](https://d-krupke.github.io/cpsat-primer/08_benchmarking.html) in the CP-SAT Primer to understand the fundamentals of benchmarking.
3. Implement the ILP formulation for graph coloring based on the paper. Do not copy the code from the paper; instead, write your own implementation. However, you may use `networkx`, as the paper does.
4. Set up a preliminary benchmarking environment to compare the performance of the different formulations. Keep in mind that solver performance may have improved since the paper was written.
5. Evaluate the preliminary results and add tests to detect inconsistencies, such as infeasible solutions or one formulation reporting a higher bound than a known solution from another formulation.
6. Implement formulations using CP-SAT and SAT-based encodings for graph coloring.
7. Compare performance across various graph generators provided by [networkx](https://networkx.org/documentation/stable/reference/generators.html).
8. Create a report summarizing the strengths and weaknesses of each formulation and solver, based on the benchmarking results.

### Possible Further Tasks

Depending on progress, we may also explore implementing heuristic approaches ourselves instead of relying exclusively on those provided by `networkx`.