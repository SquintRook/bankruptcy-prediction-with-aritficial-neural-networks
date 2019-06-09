
####### Libraries, joining datasets, data cleaning #####
setwd("C:/Users/HP/Documents/R/projekty/sieci-neuronowe/bankruty")
Sys.setenv(LANG="en")
library(foreign)
library(neuralnet)
library(nnet)
library(tidyr)
library(dplyr)
set.seed(2)

po1 <- read.arff("5year.arff")
po2 <- read.arff("4year.arff")
po3 <- read.arff("3year.arff")
po4 <- read.arff("2year.arff")
po5 <- read.arff("1year.arff")

po1 <- drop_na(po1)
po2 <- drop_na(po2)
po3 <- drop_na(po3)
po4 <- drop_na(po4)
po5 <- drop_na(po5)

po1 <- po1[2828:3031,] # bankruci sa zawsze na dole wiec mozna ?atwo wybrac ile sie chce danych
table(po1$class[2828:3031])
table(po1$class)

po2 <- po2[4530:4769,]
table(po2$class[4530:4769])

po3 <- po3[4672:4885,]
table(po3$class[4672:4885])

po4 <- po4[3943:4088,]
table(po4$class[3943:4088])

po5 <- po5[3135:3195,]
table(po5$class[3135:3194])
tail(po5$class)



suche.dane <- rbind(po1,po2,po3,po4,po5)

table(rbind(po1,po2,po3,po4,po5)[,65])
table(suche.dane$class)
niebankrut <- matrix(rep(0,nrow(suche.dane)), nrow = nrow(suche.dane), ncol = 1)

niebankrut <- suche.dane$class
niebankrut <- as.matrix(niebankrut)
niebankrut <- as.numeric(niebankrut)

niebankrut[niebankrut==0] <- 2
niebankrut[niebankrut==1] <- 0
niebankrut[niebankrut==2] <- 1

suche.dane <- cbind(suche.dane,niebankrut)
suche.dane <- drop_na(suche.dane)
########   skalowanie  #####

suche.dane<- apply(suche.dane,2, as.numeric)
maks <- apply(suche.dane, 2, max, na.rm=TRUE)
minim <- apply(suche.dane, 2, min, na.rm=TRUE)

# pozniej sproboj bez skalowania i nie z linear.output=T
scaled <- as.data.frame(scale(suche.dane, center = minim, scale = maks-minim))

czesc <- sample(1:nrow(scaled),round(0.7*nrow(scaled))) #sample wybiera probke
trening <- suche.dane[czesc,]
testowa <- suche.dane[-czesc,]
# dajemy as.data.frame bo scale daje nam macierze a ich nei chcemy
trening_ <- scaled[czesc,]
testowa_  <- scaled[-czesc,]
########### uczymy sieci#####
n <- names(trening_)
n <- n[1:64]
f <- as.formula(paste("class + niebankrut ~",
                      paste(n[!n %in% "class,niebankrut"], collapse = " + ")))
nn <- neuralnet(f,data=trening_,hidden=c(10),
                act.fct = "logistic",linear.output=FALSE,
                lifesign = "minimal",stepmax = 1e+6,threshold = 0.1,
                rep = 1, algorithm = "rprop+")

### Teraz prognozujemy naszym modelem dane testowe#####
pr.nn <- neuralnet::compute(nn, testowa_[, 1:64])
pr.nn_ <- pr.nn$net.result
head(pr.nn_)

original_values <- max.col(testowa_[,65:66])
pr.nn_2 <- max.col(pr.nn_)
mean(pr.nn_2 == original_values)



######## b??dy #######


bledy  <- as.matrix(table(Truth = original_values, Prediction = pr.nn_2))
TruPos <- bledy[1,1]
TruNeg <- bledy[2,2]
FalPos <- bledy[1,2]
FalNeg <- bledy[2,1]
TruPosRate <- TruPos/(TruPos+FalNeg)
TruNegRate <- TruNeg/(FalPos+TruNeg)
precision  <- TruPos/(TruPos+FalPos)
accuracy   <- (TruPos + TruNeg)/(length(original_values))


  tablica.bledow <- data.frame(
    TruPos,TruNeg,FalPos,FalNeg,TruPosRate,TruNegRate,accuracy)

tablica.bledow #wydruk

# TruPos - bankruty zidentyfikowane jako bankruty
# TruNeg - niebankurty zidentyfikowane jako niebankruty
# FalPos - niebankruty zidentyfikowane jako bankruty
# FalNeg - Bankruty zidentyfikowane jako niebankruty
# TruNegRate - odsetek niebankrut?w zidentyfikowanych jako niebankruty
# TruPosRate - odsetek bankrut?w zidentyfikowanych jako bankruty
# precision - precyzja lub wartos? predykcyjna dodatnia
# accuracy - to samo co wynik na poczatku
# czyli udzial poprawnie zidentyfikowanych na wszystkie obserwacje

cbind(pr.nn_2,original_values)
cor(suche.dane[,1:64],suche.dane[,65])
### teraz walidacja krzy?owa####

plot(nn)


