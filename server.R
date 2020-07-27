library(shiny)
library(dplyr)
library(tidyr)
library(ggplot2)
library(class)
library(tidyverse)
library(knitr)
library(caret)
library(ggfortify)
library(nnet)
library(DT)
# library(ggbiplot)

options(dplyr.print_min = 5)
options(tibble.print_min = 5)
opts_chunk$set(message = FALSE, cache = TRUE)
# knit_hooks$set(webgl = hook_webgl)
columns <- c("quality","fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide","total.sulfur.dioxide", "density","pH", "sulphates", "alcohol" )


set.seed(558)

shinyServer(function(input, output, session) {

  trained_model<-NA
  getData <- reactive({
    #Read the red wine data
    dataurl="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    red_wine_data=read.table(dataurl,sep = ";", header=TRUE)
    red_wine_data['wine_type'] = 'red'
    
    #Read the white wine data
    dataurl="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    white_wine_data=read.table(dataurl,sep = ";", header=TRUE)
    white_wine_data['wine_type'] = 'white'
    
    #Combine the data into one dataframe
    wine_data = union(red_wine_data, white_wine_data)
    
    #Drop rows with NA
    wine_data_ana<<-drop_na(wine_data)
    
    #Change the column of quality into levels
    wine_data_ana<<-mutate(wine_data_ana,quality_level=as.factor(quality), wine_type = as.factor(wine_type))
    
    #Standardize the data
    preproc1 <- preProcess(wine_data_ana, method=c("center", "scale"))
    wine_data_norm <<- predict(preproc1, wine_data_ana)
    wine_data_norm['quality'] <<- wine_data_ana['quality']
    
    #If we want to keep all the data, then don't apply filter, otherwise, apply filter
    if(input$wine_type =="all"){
      newData <- wine_data_norm
    }else{
      newData <- wine_data_norm %>% filter(wine_type == input$wine_type)
    }
  })
  
  # Use the PCA method to get the components
  getPcaResult<-reactive({
    pca_vars<-input$PcaSelectedColumns
    wine_data_ana_pca<-wine_data_norm
    
    PCs<-prcomp(wine_data_ana_pca[,pca_vars])
    return(list(pcas=PCs))
  })
  
  PCA.results<-reactive({
    t<-getPcaResult()
    t$pcas}
  )
  
  # Run supervised methods.
  getSupervisedResult<-reactive({
    
    set.seed(100)
    pct=input$TrainingPercent/100.0
    neighbors=input$NeighborCount
    predictors= input$SelectedColumns
    
    train_idx <- sample(1:nrow(wine_data_ana), floor(pct*nrow(wine_data_ana)))
    

    train_wine_data_ana <- wine_data_norm[train_idx, ]
    test_wine_data_ana <- wine_data_norm[-train_idx, ]
    
    response_var <- 'quality_level'
    if(input$SupervisedMethod=='knn (quality as factor)'){
      t<-train_wine_data_ana[,c(response_var,predictors)]
      test<- test_wine_data_ana[,c(response_var,predictors)]
      ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
      #Make the model accessible to other function
      trained_model <<- train(quality_level ~ ., data = t, method = "knn", trControl = ctrl,  tuneLength = 5)
      fitted.values<-predict(trained_model, newdata= test)

    }
    else{
      if(input$SupervisedMethod=='multinomial (quality as factor)'){
        #Use the multinomial log-linear models via neural networks.
        train_wine_data_tmp<-train_wine_data_ana[,c('quality_level',predictors)]
        test <- multinom(quality_level ~ ., data = train_wine_data_tmp)
        fitted.values<-predict(test,newdata=test_wine_data_ana)
        #Make the model accessible to other function
        trained_model <<- test
      }
      else{
        if(input$SupervisedMethod=='Ridge regression (quality as numeric)'){
          response_var <- 'quality'
          #5 fold CV for all relevant models
          control_cv <- trainControl(method = 'cv', number = 5)
          lambda_grid <- expand.grid(lambda = 10 ^ seq(-2, 3, .1))

          
          ridge <- train(quality ~ ., method="ridge", subset=train_idx, data=wine_data_norm[,c(response_var,columns)],
                         trControl = control_cv,
                         tuneGrid = lambda_grid)
          
          #Make the model accessible to other function
          trained_model <- ridge
          #need to figure out how to get model coeffs. Just fit new ridge with besttune? Probably?
          fitted.values <- predict(ridge, newdata=test_wine_data_ana[,c(response_var,columns)])
          test_mse<-mean((fitted.values - test_wine_data_ana$quality)^2)
          result_desc<-paste('The Mean Square Error (MSE) for the training dataset is ', ridge$results$RMSE[1], '<br>and the test MSE is ', test_mse)
        }
      }
    }
    
    # Get the confusion matrix for the test results
    if(response_var=='quality_level'){
      fitInfo <- tbl_df(data.frame(predicted=fitted.values, test_wine_data_ana[, c(response_var,predictors)]))
      contingencyTbl<-table(fitInfo$predicted,fitInfo$quality_level)
      misClass <- 1 - sum(diag(contingencyTbl))/sum(contingencyTbl)
      result_desc <- paste('Above shows the confusion matrix for the classification of testing data. <br>The misclassification percentage for the testing data is ', round(misClass*100,2))
      ret_list <-list(contingencyTbl=as.data.frame.matrix(contingencyTbl), result_desc=result_desc)
    }
    else{
      ret_list <-list(contingencyTbl=NA, result_desc=result_desc)
    }
  })
  
  observeEvent(input$predict_click, {
    new_wine <- data.frame(fixed.acidity=input$fixed.acidity, volatile.acidity=input$volatile.acidity, citric.acid=input$citric.acid, residual.sugar=input$residual.sugar,
                           chlorides=input$chlorides, free.sulfur.dioxide=input$free.sulfur.dioxide, total.sulfur.dioxide=input$total.sulfur.dioxide, density=input$density,
                           pH=input$pH, sulphates=input$sulphates, alcohol=input$alcohol)
    if(is.na(trained_model)){
      output$predicted_quality<-renderText({print("<b style='color:#0000FF'>There is no trained model yet.</b>")})
    }
    else{
      predicted_respose<-predict(trained_model, newdata=new_wine)
      output$predicted_quality<-renderText({print(paste("<b style='color:#0000FF'>The predicted wine quality based on your input is:",predicted_respose[[1]], "</b>"))})
    }
  })
  
  plotInput <- reactive({
    newData <- getData()
    
    #create plot
    g <- ggplot(newData, aes(x = quality_level, y = citric.acid))
    if(input$cb_wine_type){
      g + geom_point() + geom_jitter(aes(colour = wine_type))
    } else {
      g + geom_boxplot() + geom_jitter()
    }
  })
  
  
  
  #create plot
  output$wine_dataPlot <- renderPlot({
    plotInput()
  })
  
  
  #create output of observations    
  output$table <- DT::renderDataTable(
    DT::datatable(getData(), options = list(pageLength = 10))%>% formatRound(columns = c(1:11), digits = 2)
  )
  
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("wine_data_",input$wine_type, ".csv", sep = "")
    },
    content = function(file) {
      write.csv(getData(), file, row.names = FALSE)
    }
  )
  
  output$downloadPlot <- downloadHandler(
    filename = function() { paste("wine_data_",input$wine_type, '.png', sep='') },
    content = function(file) {
      ggsave(file,plotInput())
    }
  )
  
  
  output$plot_clicked_points <- DT::renderDataTable({
    # With base graphics, we need to explicitly tell it which variables were
    # used; with ggplot2, we don't.
    res <- nearPoints(getData(), input$plot_click, threshold = 5, maxpoints = 10, addDist = TRUE) 

    datatable(res)%>% formatRound(columns = c(1:11), digits = 2)
  })
  
  output$brush_info <- DT::renderDataTable({
    res <-  brushedPoints(getData(), input$plot1_brush)
    datatable(res)%>% formatRound(columns = c(1:11), digits = 2)
  })
  
  
  output$knn_output<-renderTable({
    rslt<-getSupervisedResult()
    rslt$contingencyTbl
  }, include.rownames=TRUE)
  
  output$misclassifiedPct<-renderText({
    rslt<-getSupervisedResult()
    print(paste("By using the ", input$SupervisedMethod, "I got the result of ", rslt$result_desc ))
  })
  
  
  .theme<- theme(
    axis.line = element_line(colour = 'gray', size = .75), 
    panel.background = element_blank(),  
    plot.background = element_blank()
  )	
  
  #make componentplot
  output$componentplot <- renderPlot({
    #autoplot(getPcaResult()$pcas, data = wine_data_ana, colour = 'wine_data', loadings = TRUE,loadings.label = TRUE)
    PCs<-getPcaResult()$pcas
    st<-min(as.integer(input$NumberOfPcs),length(PCs)-1)
    autoplot(PCs, data = wine_data_ana, colour = 'quality', loadings = TRUE)
    # ggbiplot(PCs,ellipse=TRUE,  labels=rownames(wine_data_ana), groups=wine_data_ana$quality_level)
    
    # ggbiplot(PCs, obs.scale = 1, var.scale = 1, ellipse = input$plotEllipse, choices=st:(st+1), circle = input$plotCircle, var.axes=TRUE, labels=c(wine_data_ana[,"quality_level"]), groups=as.factor(c(wine_data_ana[,"quality_level"])))
    
  })
  
  output$screeplot <- renderPlot({
    pca_output <- getPcaResult()$pcas
    eig = (pca_output$sdev)^2
    variance <- eig*100/sum(eig)
    cumvar <- paste(round(cumsum(variance),1), "%")
    eig_df <- data.frame(eig = eig,
                         PCs = colnames(pca_output$x),
                         cumvar =  cumvar)
    ggplot(eig_df, aes(reorder(PCs, -eig), eig)) +
      geom_bar(stat = "identity", fill = "white", colour = "black") +
      geom_text(label = cumvar, size = 4,
                vjust=-0.4) +
      theme_bw(base_size = 14) +
      xlab("PC") +
      ylab("Variances") +
      ylim(0,(max(eig_df$eig) * 1.1))
  })
})