rm(list = ls())
library("mlbench")
datos <- read.csv("data/filtered_data.csv", header = T)
dim(datos)

library(caret)
y <- datos$ln_years_street
df <- data.frame(
  "gender" = datos$gender, # var2
  "race_minority" = datos$race_minority, # var3
  "lgbt_minority" = datos$lgbt_minority, # var4
  "disability" = datos$disability, # var5
  "disease" = datos$disease, # var6
  "family_contact" = datos$family_contact, # var7
  "recieves_help" = datos$recieves_help, # var8
  "years_education" = datos$years_education, # var9
  "drug_consumption" = datos$drug_consumption, # var10
  "avg_age_drug_consumption" = datos$avg_age_drug_consumption, # var11
  "age" = datos$age # var12
)
set.seed(42)
test_inds = createDataPartition(y = 1:length(y), p = 0.2, list = F)

df_test = df[test_inds, ]
y_test = y[test_inds]
df_train = df[-test_inds, ]
y_train = y[-test_inds]

paste("Shape of the train data: ")
paste(dim(df_train))
paste("Shape of the test data: ")
paste(dim(df_test))

options(java.parameters="-Xmx5000m")
library(bartMachine)
bart_machine = bartMachine(df_train, y_train)
summary(bart_machine)

rmse_by_num_trees(bart_machine,
                  tree_list=c(seq(5, 75, by=5)),
                  num_replicates=3)

bart_machine <- bartMachine(df_train, y_train, num_trees=40, seed=42)
plot_convergence_diagnostics(bart_machine)

check_bart_error_assumptions(bart_machine)


plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)
plot_y_vs_yhat(bart_machine, Xtest=df_test, ytest=y_test, prediction_intervals = TRUE)


rmse <- function(x, y) sqrt(mean((x - y)^2))
rsq <- function(x, y) summary(lm(y~x))$r.squared
y_pred <- predict(bart_machine, df_test)
paste('r2:', rsq(y_test, y_pred))
paste('rmse:', rmse(y_test, y_pred))
cor.test(y_test, y_pred, method=c("pearson"))

investigate_var_importance(bart_machine, num_replicates_for_avg = 20)
