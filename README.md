# Predicting-Customer-Response
Did this project for Data Science. These data are a subset of data from a targeted marketing campaign.  Each data record  describes a targeted consumer by various attributes. Had to answer the following questions-
a)What is the structure of the file?  What are the attributes/fields?  How many customers are there?     
b)Do any two customers have the same name?  If so, list the duplicated name(s).   
c)Sometimes data values are missing and programs in the data  processing pipeline fill in special values that can screw up analysis.  How many rows have value -9999?  Do not consider them in any further  processing.   
d)RFA_2F and RFA_2A are marketing codes summarizing the recency and  frequency of historical customer response.  What values do they take  and what is the frequency of each value in the data?  What values does  WEALTH_INDEX take?   
e)TARGET_B describes whether this customer responded (1) or did not  respond (0) to the campaign.  What proportion of the targeted  consumers responded?   
f)Create a dendrogram. How many clusters are evident? Justify your answer.   
g)Plot the distributions of WEALTH_INDEX for the responders and  non-responders on one graph to show the similarities and differences.  Use a histogram.  What is the difference in the graphs from plotting  percents versus counts on the y-axis?
h)Alphabetize the records by the NAME field. Number them from 1 at  the beginning of the alphabet. Do not print them all out.Just print  out alphabetized records 1-10 and alphabetized records 20,000-20,010.   
i)Redo part (h), except alphabetize by the individual's *last name*.   
j)Split the data into an appropriate Training and Test set. Using the Training data, train 3 predictive models (choose 3 appropriate classifiers) for TARGET_B (whether the customer responds). Using the Test set, compare their performance using ROC plots and Area Under the Curves (AUC). Which is the "best" according to your analysis?
