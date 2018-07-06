## Hiring Challenge for Data Scientists

### Introduction

Two of our clients have a similar problem: they each have data on which of their customers made purchases at which times, and they each want to assign a value to each of their customers (For different reasons -- client A wants to target an email campaign at customers that are at risk of never purchasing again, while client B wants to quantify customer health over time). Your task is to develop a model that can take either client's data, and returns a "health" score for each customer.

### Instructions

Build a standalone command-line application in the language of your choice that takes an input datset of customer transaction data and, when run:
  - trains a model that predicts a customer's health as a float from `0.0 - 1.0` given their transaction history, and
  - prints a csv containing on each row a customer ID and that customer's health score.

The program needs to be able to ingest both provided files as input, and print the output on stdout.

You can use any library functions you like, or build a model from scratch if you prefer.

There is no right answer as such, we will mainly be assessing code quality, data preprocessing skills and clarity of thought.

### Files

- `input_data.csv` - sample transaction data to use as input to geernate customer predictions
