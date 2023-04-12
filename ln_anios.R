rm(list = ls())
Data <- read.csv("/Users/rmateusc/Documents/9/Econometria Bayesiana/trabajo final/filtered_data.csv", header = T)
attach(Data)
y <- ln_anios_en_calle
X <- cbind(
  1, 
  hombre, # var2
  minoria_raza, # var3
  minoria_lgbt, # var4
  discapacidad, # var5
  enfermedad, # var6
  contacto_familia, # var7 
  recibe_ayuda, # var8
  anios_educacion, # var9
  consume_drogas, # var10
  edad_promedio_inicio_consumo, # var11
  edad # var12
  )
k <- dim(X)[2]
N <- dim(X)[1]

### Interpretacion Resultados
# var2: ser hombre 12.95% mas anios en la calle 
# var3: ser minoria raza 16.37% mas anios en la calle
# var4: no es estadisticamente relevante
# var5: no es estadisticamente relevante
# var6: no es estadisticamente relevante
# var7: no tener contacto con familia 15.04% mas anios en la calle
# var8: no es estadisticamente relevante
# var9: tener bachillerato 24.85% mas de anios en la calle
# var10: si consume drogas actualmente 94.26% anios en en la calle
# var11: un anio mas de madurez en el inicio de consumo de drogas 1.80% menos de anios en la calle
# var12: un anio mas de edad 3.14% mas de anios en la calle


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

