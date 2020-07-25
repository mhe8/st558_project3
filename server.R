library(shiny)
library(dplyr)
library(tidyr)
library(ggplot2)
library(class)
library(tidyverse)
library(knitr)
# library(rgl)
library(ggfortify)
# library(ggbiplot)

options(dplyr.print_min = 5)
options(tibble.print_min = 5)
opts_chunk$set(message = FALSE, cache = TRUE)
# knit_hooks$set(webgl = hook_webgl)

shinyServer(function(input, output, session) {
  getData <- reactive({
    
    dataurl="http://www.statsci.org/data/oz/ctsibuni.txt"
    ctsib=read.table(dataurl,sep="\t",header=TRUE)
    
    ctsib_ana<-ctsib%>%filter(Age!="NA"&Height!="NA"&Weight!="NA"&Surface!="NA"&Vision!="NA")
    
    ctsib_ana<-mutate(ctsib_ana,CTSIB=as.factor(CTSIB))
    ctsib_ana$Sex= as.integer(as.factor(ctsib_ana$Sex))
    ctsib_ana$Surface= as.integer(as.factor(ctsib_ana$Surface))
    ctsib_ana$Vision= as.integer(as.factor(ctsib_ana$Vision))
    ctsib_ana<-mutate(ctsib_ana,BMI=Weight/(Height/100.0)^2)
    ctsib_ana<-mutate(ctsib_ana,PI=1000*Weight^(1/3)/Height)
    
    
    if(input$Sex =="all"){
      newData <- ctsib %>% filter(Surface == input$Surface & Vision==input$Vision)
    }else{
      newData <- ctsib %>% filter(Surface == input$Surface & Vision==input$Vision & Sex == input$Sex)
    }
  })
  
  getPcaResult<-reactive({
    pca_vars<-input$PcaSelectedColumns
    ctsib_ana_pca<-ctsib_ana
    for(p in predictors){
      colm<-colMeans(ctsib_ana_pca[p])
      colsd<-sqrt(sum((ctsib_ana_pca[p]-colm)^2)/(nrow(ctsib_ana_pca)-1))
      ctsib_ana_pca[,p]<-(ctsib_ana_pca[,p]-colm)/colsd
    }
    PCs<-prcomp(ctsib_ana_pca[,pca_vars], scale=T)
    return(list(pcas=PCs))
  })
  
  PCA.results<-reactive({
    t<-getPcaResult()
    t$pcas})
  
  getSupervisedResult<-reactive({
    
    set.seed(100)
    pct=input$TrainingPercent/100.0
    neighbors=input$NeighborCount
    predictors= input$SelectedColumns
    print(input$SelectedColumns)
    subjects<-unique(ctsib_ana$Subject)
    
    train_sample_size<-floor(pct*length(subjects))
    train_subj<-sample(subjects,train_sample_size )
    train_ctsib_ana<-filter(ctsib_ana, Subject %in% train_subj)
    test_ctsib_ana<-filter(ctsib_ana, !(Subject %in% train_subj))
    
    
    for(p in c("Height","Weight","BMI","PI")){
      colm<-colMeans(train_ctsib_ana[p])
      colsd<-sqrt(sum((train_ctsib_ana[p]-colm)^2)/(nrow(train_ctsib_ana)-1))
      train_ctsib_ana[,p]<-(train_ctsib_ana[,p]-colm)/colsd
      test_ctsib_ana[,p]<-(test_ctsib_ana[,p]-colm)/colsd
    }
    
    if(input$SupervisedMethod=='knn'){
      fitted.values <- knn(train = train_ctsib_ana[,predictors],
                           test = test_ctsib_ana[,predictors],
                           cl = train_ctsib_ana$CTSIB,
                           k = neighbors, prob=TRUE) #could use CV to determine k
    }
    else{
      train_ctsib_tmp<-train_ctsib_ana[,c('CTSIB',predictors)]
      test <- multinom(CTSIB ~ ., data = train_ctsib_tmp)
      fitted.values<-predict(test,newdata=test_ctsib_ana)
    }
    
    fitInfo <- tbl_df(data.frame(predicted=fitted.values, test_ctsib_ana[, c("CTSIB",predictors)]))
    contingencyTbl<-table(fitInfo$predicted,fitInfo$CTSIB)
    misClass <- 1 - sum(diag(contingencyTbl))/sum(contingencyTbl)
    
    return(list(contingencyTbl=as.data.frame.matrix(contingencyTbl), misclassified=misClass))
  })
  
  
  plotInput <- reactive({
    newData <- getData()
    
    #create plot
    g <- ggplot(newData, aes(x = CTSIB, y = Weight))
    if(input$conservation){
      g + geom_point(size = input$size, aes(col = Sex))
    } else {
      g + geom_point(size = input$size)
    }
  })
  
  #create plot
  output$ctsibPlot <- renderPlot({
    plotInput()
  })
  
  
  #create output of observations    
  output$table <- renderTable({
    getData()
  })
  
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("CTSIB_",input$Vision, "_", input$Surface, ".csv", sep = "")
    },
    content = function(file) {
      write.csv(getData(), file, row.names = FALSE)
    }
  )
  
  output$downloadPlot <- downloadHandler(
    filename = function() { paste("CTSIB_",input$Vision, "_", input$Surface, '.png', sep='') },
    content = function(file) {
      ggsave(file,plotInput())
    }
  )
  
  
  output$click_info <- renderPrint({
    # Because it's a ggplot2, we don't need to supply xvar or yvar; if this
    # were a base graphics plot, we'd need those.
    nearPoints(getData(), input$plot1_click, addDist = TRUE)
  })
  
  output$brush_info <- renderPrint({
    brushedPoints(getData(), input$plot1_brush)
  })
  
  
  output$knn_output<-renderTable({
    rslt<-getSupervisedResult()
    rslt$contingencyTbl
  }, include.rownames=TRUE)
  
  output$misclassifiedPct<-renderText({
    rslt<-getSupervisedResult()
    print(paste0("Miss classified percentage: ",rslt$misclassified*100.0))
  })
  
  
  .theme<- theme(
    axis.line = element_line(colour = 'gray', size = .75), 
    panel.background = element_blank(),  
    plot.background = element_blank()
  )	
  
  
  #file info	
  output$filetable <- renderTable({
    tmp<-ctsib_ana
    tmp<-tmp[,seq_along(1:ncol(tmp))<=10] # show max 10 columns and binf head tail calls
    rbind(head(tmp,10),tail(tmp,10))
    
  })
  
  
  #make componentplot
  output$componentplot <- renderPlot({
    #autoplot(getPcaResult()$pcas, data = ctsib_ana, colour = 'CTSIB', loadings = TRUE,loadings.label = TRUE)
    PCs<-getPcaResult()$pcas
    st<-min(as.integer(input$NumberOfPcs),length(PCs)-1)
    
    ggbiplot(PCs, obs.scale = 1, var.scale = 1, ellipse = input$plotEllipse, choices=st:(st+1), circle = input$plotCircle, var.axes=TRUE, labels=c(ctsib_ana[,"CTSIB"]), groups=as.factor(c(ctsib_ana[,"CTSIB"])))
    
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