if (!require(bsts)) {
  install.packages('bsts')
  library(bsts)
}

####### Data loading ##########
X_r1 <- read.csv("X_r1.csv")
Y_r1 <- read.csv("Y_r1.csv")
train_X <- X_r1[1:11592,]
train_Y <- Y_r1[1:11592,]
tail(train_X)
test_X <- tail(X_r1, 24)
test_Y <- tail(Y_r1, 24)
typeof(train_Y)

####### Univariate #######
y <- as.numeric(train_Y)
solargen_ss <- AddLocalLinearTrend(list(), y)
solargen_ss <- AddSeasonal(solargen_ss, y, nseasons = 24)
solargen.bayesian.model <- bsts(y, state.specification = solargen_ss,
                                niter = 1000)
solargen.burn <- SuggestBurn(0.1, solargen.bayesian.model)
solargen.horizon.pred <- 24

solargen.bayesian.pred <- predict.bsts(solargen.bayesian.model,
                                       horizon = solargen.horizon.pred,
                                       burn = solargen.burn,
                                       quatiles = c(.025, .975))
####### Multivariate ######
multi.solargen_ss <- AddLocalLinearTrend(list(), y)
multi.solargen_ss <- AddSeasonal(multi.solargen_ss, y, nseasons = 24)
multi.solargen_ss.bayesian.model <- bsts(y ~., state.specification = multi.solargen_ss,
                                         data = train_X, niter = 500)
multi.solargen.burn <- SuggestBurn(0.1, multi.solargen_ss.bayesian.model)
multi.solargen.horizon.pred <- 24

multi.solargen.bayesian.pred <- predict.bsts(multi.solargen_ss.bayesian.model,
                                       horizon = multi.solargen.horizon.pred,
                                       burn = multi.solargen.burn,
                                       newdata = test_X,
                                       quatiles = c(.025, .975))
multi.solargen.bayesian.pred$original.series

coef <- multi.solargen_ss.bayesian.model$coefficients
coef <- coef[-c(1:50), ]
apply(coef, 2, median)

library(lubridate)
library(ggplot2)
library(tidyverse)
library(magrittr)

X_r1 %<>%
  mutate(time = make_datetime(year = year, month = month, day = day, hour = hour))
X_r1[['year']] <- ifelse(is.na(X_r1[['year']]), 2023, X_r1[['year']])

### univariate data frame ###
solargen.bayesian.df <- data.frame(
  c(train_X$time, seq(max(train_X$time) + hours(1), by = 'hours', length.out = solargen.horizon.pred)),
  c(as.numeric(train_Y), rep(NA, solargen.horizon.pred)),
  c(as.numeric(-colMeans(solargen.bayesian.model$one.step.prediction.errors[-(1:solargen.burn), ])+y),
               as.numeric(solargen.bayesian.pred$mean))
)

names(solargen.bayesian.df) <- c('Date', 'Actual', "Fitted")

### multivariate data frame ###
multi.solargen.bayesian.df <- data.frame(
  c(train_X$time, seq(max(train_X$time) + hours(1), by = 'hours', length.out = multi.solargen.horizon.pred)),
  c(as.numeric(train_Y), rep(NA, multi.solargen.horizon.pred)),
  c(as.numeric(-colMeans(multi.solargen_ss.bayesian.model$one.step.prediction.errors[-(1:multi.solargen.burn), ])+y),
    as.numeric(multi.solargen.bayesian.pred$mean))
)

names(multi.solargen.bayesian.df) <- c('Date', 'Actual', "Fitted")

### univariate score ###
solargen.bayesian.df %>% 
  filter(is.na(Actual) == F) %>%
  summarise(MAE=mean(abs(Actual-Fitted))) -> train_MAE
solargen.bayesian.df %>%
  filter(is.na(Actual) == T) %>%
  summarise(MAE=mean(abs(test_Y$amount-Fitted))) -> test_MAE

### multivariate score ###
multi.solargen.bayesian.df %>% 
  filter(is.na(Actual) == F) %>%
  summarise(MAE=mean(abs(Actual-Fitted))) -> train_MAE
multi.solargen.bayesian.df %>%
  filter(is.na(Actual) == T) %>%
  summarise(MAE=mean(abs(test_Y$amount-Fitted))) -> test_MAE


solargen.bayesian.df %>%
  filter(is.na(Actual) == T) %>%
  mutate(Actual = test_Y$amount) -> solargen.bayesian.df.test.pred

multi.solargen.bayesian.df %>%
  filter(is.na(Actual) == T) %>%
  mutate(Actual = test_Y$amount) -> multi.solargen.bayesian.df.test.pred

solargen.bayesian.posterior.interval <- data.frame(
  filter(solargen.bayesian.df, is.na(Actual) == T)[, 1],
  solargen.bayesian.pred$interval[1,],
  solargen.bayesian.pred$interval[2,]
)
names(solargen.bayesian.posterior.interval) <- c("Date", "LL", "UL")

multi.solargen.bayesian.posterior.interval <- data.frame(
  filter(multi.solargen.bayesian.df, is.na(Actual) == T)[, 1],
  multi.solargen.bayesian.pred$interval[1,],
  multi.solargen.bayesian.pred$interval[2,]
)

names(multi.solargen.bayesian.posterior.interval) <- c("Date", "LL", "UL")


#### Train Plot ####
solargen.bayesian.df.pred <- left_join(solargen.bayesian.df, solargen.bayesian.posterior.interval, by="Date")
multi.solargen.bayesian.df.pred <- left_join(multi.solargen.bayesian.df, multi.solargen.bayesian.posterior.interval, by="Date")

solargen.bayesian.df.pred %>%
  ggplot(aes(x=Date)) +
  geom_line(aes(y=Actual, colour = "Actual"), size=1.2) +
  geom_line(aes(y=Fitted, colour = "Fitted"), size=1.2, linetype=2) +
  theme_bw() + theme(legend.title = element_blank()) + ylab("") + xlab("") +
  geom_vline(xintercept=as.numeric(as.Date("2020-01-01")), linetype=2) + 
  geom_ribbon(aes(ymin=LL, ymax=UL), fill="grey", alpha=0.5) +
  ggtitle(paste0("BSTS -- Holdout MAPE = ", round(100*MAPE,2), "%")) +
  theme(axis.text.x=element_text(angle = -90, hjust = 0))

multi.solargen.bayesian.df.pred %>%
  ggplot(aes(x=Date)) +
  geom_line(aes(y=Actual, colour = "Actual"), size=1.2) +
  geom_line(aes(y=Fitted, colour = "Fitted"), size=1.2, linetype=2) +
  theme_bw() + theme(legend.title = element_blank()) + ylab("") + xlab("") +
  geom_vline(xintercept=as.numeric(as.Date("2020-01-01")), linetype=2) + 
  geom_ribbon(aes(ymin=LL, ymax=UL), fill="grey", alpha=0.5) +
  ggtitle(paste0("BSTS -- Holdout MAPE = ", round(100*MAPE,2), "%")) +
  theme(axis.text.x=element_text(angle = -90, hjust = 0))

#### Test Plot ####
solargen.bayesian.df.test.pred <- left_join(solargen.bayesian.df.test.pred, solargen.bayesian.posterior.interval, by="Date")
solargen.bayesian.df.test.pred
solargen.bayesian.df.test.pred %>%
  ggplot(aes(x=Date)) +
  geom_line(aes(y=Actual, colour = "Actual"), size=1.2) +
  geom_line(aes(y=Fitted, colour = "Fitted"), size=1.2, linetype=2) +
  theme_bw() + theme(legend.title = element_blank()) + ylab("") + xlab("") +
  geom_vline(xintercept=as.numeric(as.Date("2023-10-15")), linetype=2) + 
  geom_ribbon(aes(ymin=LL, ymax=UL), fill="grey", alpha=0.5) +
  ggtitle(paste0("BSTS -- Holdout MAE = ", round(100*test_MAE,2), "%")) +
  theme(axis.text.x=element_text(angle = -90, hjust = 0))

multi.solargen.bayesian.df.test.pred <- left_join(multi.solargen.bayesian.df.test.pred, 
                                                  multi.solargen.bayesian.posterior.interval, by="Date")
multi.solargen.bayesian.df.test.pred
multi.solargen.bayesian.df.test.pred %>%
  ggplot(aes(x=Date)) +
  geom_line(aes(y=Actual, colour = "Actual"), size=1.2) +
  geom_line(aes(y=Fitted, colour = "Fitted"), size=1.2, linetype=2) +
  theme_bw() + theme(legend.title = element_blank()) + ylab("") + xlab("") +
  geom_vline(xintercept=as.numeric(as.Date("2023-10-15")), linetype=2) + 
  geom_ribbon(aes(ymin=LL, ymax=UL), fill="grey", alpha=0.5) +
  ggtitle(paste0("BSTS -- Holdout MAE = ", round(100*test_MAE,2), "%")) +
  theme(axis.text.x=element_text(angle = -90, hjust = 0))
