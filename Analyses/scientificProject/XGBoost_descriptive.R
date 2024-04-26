# summarize descriptives 
library('knitr')
library('tidyverse')
library('emmeans')
library('afex')
path <- 'Dropbox/UTcourses/machine_learning/output.csv'
dat <- read.csv(path)

# get mean of each each subject
apply(dat, 1, mean) %>%
        round(digits = 3)
apply(dat, 2, mean) %>%
        round(digits = 3)
dat_long <- dat %>%
        pivot_longer(names_to = 'condition', values_to = 'AUC', cols = Maintain:Suppress)
result <- aov_car(AUC ~ condition + Error(sub/condition), data = dat_long)
emmeans(result, ~ condition)
confint(emmeans(result, pairwise ~ condition, contrast = 'consec'))

result <- aov_car(AUC ~ sub + Error(condition/sub), data = dat_long)
emmeans(result, ~ sub)
emmeans(result, pairwise ~ sub, contrast = 'consec')

