rm(list = ls())
datos <- read.csv("data/filtered_data.csv", header = T)
# attach(datos)
y <- datos$ln_years_street
X <- cbind(
  1,
  datos$gender, # var2
  datos$race_minority, # var3
  datos$lgbt_minority, # var4
  datos$disability, # var5
  datos$disease, # var6
  datos$family_contact, # var7
  datos$recieves_help, # var8
  datos$years_education, # var9
  datos$drug_consumption, # var10
  datos$avg_age_drug_consumption, # var11
  datos$age # var12
  )

k <- dim(X)[2]
N <- dim(X)[1]

###############Normal-Inverse Gamma Modelo: Indepedent Priors
#####Gibbs sampler########
# Hyperparameters
B0<-1000*diag(k) #Prior covariance Matrix Normal distribution
B0i<-solve(B0) #Prior precision Matrix Normal distribution
b0<-rep(0,k) #Prior mean Normal distribution
a0<-0.001 #Prior shape parameter Inverse-Gamma distribution
d0<-0.001  #Prior rate parameter Inverse-Gamma distribution

library(LearnBayes)
burn<-5000
it <- 10000
tot<-burn+it
betaGibbs<-matrix(0,tot,k)
varGibbs<-matrix(1,tot,1)
sigma2_0<-10
Bp<-solve(B0i+sigma2_0^{-1}*t(X)%*%X)
bp<-Bp%*%(B0i%*%b0+sigma2_0^{-1}*t(X)%*%y)
ap<-a0+N
for(i in 1:tot){
  BetaG<-MASS::mvrnorm(1, mu=bp, Sigma=Bp)
  dp<-d0+t(y-X%*%BetaG)%*%(y-X%*%BetaG)
  sigma2<-rigamma(1,ap/2,dp/2) #Variance parameter
  Bp<-solve(B0i+sigma2^{-1}*t(X)%*%X)
  bp<-Bp%*%(B0i%*%b0+sigma2^{-1}*t(X)%*%y)
  betaGibbs[i,]<-BetaG
  varGibbs[i,]<-sigma2
}
betasG<-coda::mcmc(betaGibbs[burn:it,])
varsG<-coda::mcmc(varGibbs[burn:it,])
library(coda)
summary(betasG)
summary(varsG)
par(mar = c(1, 1, 1, 1))
plot(betasG)
plot(varsG)
autocorr.plot(betasG)
autocorr.plot(varsG)
geweke.diag(betasG)
geweke.diag(varsG)
raftery.diag(betasG,q=0.5,r=0.025,s = 0.95)
raftery.diag(varsG,q=0.5,r=0.025,s = 0.95)
heidel.diag(betasG)
heidel.diag(varsG)
