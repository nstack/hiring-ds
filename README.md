## Hiring Challenge for Data Scientists

### Introduction

Two of our "clients" have a similar problem: they each have data on which of their customers made purchases at which times, and they each want to assign a value to each of their customers (there could be different commercial reasons for this -- for instance you may want to use the information to target an email campaign at customers that are at risk of never purchasing again, or simply to quantify customer health over time).
Your task is to build a solution that can take either client's data, and returns a "health" score for each customer.

### Instructions

Build a standalone command-line application in the language of your choice that takes the path of a single input CSV datafile as a command-line argument and
  - loads and validates the input dataset of customer transaction data
  - trains a model that predicts a customer's health as a float from `0.0 - 1.0` given their transaction history
  - prints a CSV file containing the customer ID and health score per row to `stdout`

The `orders.zip` archive contains two sample transaction datasets that can each be used as input to generate customer predictions. The files come from two different domains and are independent, with their own schema and consist of "messy" real world data - your solution is expected to be able to work with each sample dataset individually to output predictions.

You can use any 1st- or 3rd-party existing library functions, packages, frameworks, models, and solvers you like, or can build a solution/model from scratch if you prefer.

There is no right answer as such, we will mainly be looking at code quality, data preprocessing skills, completeness of the solution from a software engineering perspective, and clarity of thought.

To get started, we recommend forking and cloning this repo, and then either point us to your fork or submit a PR - thanks! (Note, you'll need `git-lfs` installed to pull down the datasets, or just download direct from the [GitHub source browser](https://github.com/nstack/hiring-ds/blob/master/orders.zip))
