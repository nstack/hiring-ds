## Hiring Challenge for Data Scientists

### Introduction

Two of our "clients" have a similar problem: they each have data on which of their customers made purchases at which times, and they each want to assign a value to each of their customers (there could be different commercial reasons for this -- for instance you may want to use the information to target an email campaign at customers that are at risk of never purchasing again, or simply to quantify customer health over time).
Your task is to develop a model that can take either client's data, and returns a "health" score for each customer.

### Instructions

Build a standalone command-line application in the language of your choice that takes the path of a input CSV datafile as a command-line argument and
  - loads and validates the input dataset of customer transaction data
  - trains a model that predicts a customer's health as a float from `0.0 - 1.0` given their transaction history
  - prints a CSV file containing the customer ID and health score per row to `stdout`

You can use any 1st- or 3rd-party library functions, packages, frameworks and solvers you like, or build a model from scratch if you prefer.

There is no right answer as such, we will mainly be looking at code quality, data preprocessing skills, completeness of the solution from a software engineering perspective, and clarity of thought.

### Files

- `orders.zip` - an archive containing two sample transaction datasets to use as input to generate customer predictions
