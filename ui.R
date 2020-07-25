library(ggplot2)

library(class)
dataurl="http://www.statsci.org/data/oz/ctsibuni.txt"
ctsib=read.table("http://www.statsci.org/data/oz/ctsibuni.txt",sep="\t",header=TRUE)


shinyUI(navbarPage("Project 3 - Min He",
                   tabPanel("Projects requirements",
                            mainPanel(
                              HTML("
<h2 style='color:#009933'>Requirements from Project 3</h2>
     
<h3>Creating a Shiny App</h3>
<p>The goal of this project is to create a nice looking shiny app that can be used to explore data and model it.</p>
<h3>Project Work</h3>
<p>
All project work should be done in a github repo. Ideally, you have connected RStudio with github and can work from the command line within RStudio. All major updates should be made through github so we can track your activity.</p>
<p>
We will run your app from gitHub using RStudio so make sure it is set up correctly to do that (see lecture for reference to setting it up, try it yourself to make sure it works!).</p>
<p>You should simply post your github repo URL as your submission.</p>
</p>
<h3>Find a data set you are interested in</h3>
<p>For this project I'm going to let you choose your own data set. This should be a data set that has at least two categorical and at least two quantitive variables. See the previous project for a list of places with datasets. Alternatively, you could pull data from the web within your app (for instance baseball data (<a href=https://pitchrx.cpsievert.me/>link</a>) or twitter data (<a href=https://cran.r-project.org/web/packages/rtweet/vignettes/intro.html>link</a>)). Choose something you are interested in and have ideas for investigating!</p>
<h3>App Requirements</h3>
<ul>
    <li>Your app should have multiple pages (tabs) to it. I don’t care if you use the built in tabs for shiny or a package like shinydashboard - use the method you prefer.
      <ul>
        <li>An information page that describes the data and abilities of the app</li>
        <li>A data exploration page where common numeric and graphical summaries can be created by the user</li>
        <li>A page with either clustering (include a dendogram) or principal components analysis (include a biplot) - again where the user can specify aspects of the algorithm</li>
        <li>A page for modeling - see below for details</li>
        <li>A page that allows the user to scroll through the data (or subset of data of interest)</li>
      </ul>
    </li>
    <li> You should have at least two different dynamic UI elements.</li>
    <li> You should have a button that allows the user to save a plot they are viewing to a file.</li>
    <li> You should have an option to allow the user to save the data currently being used for a plot (or when they are looking at the data table) to a .csv (or other) file.</li>
    <li> You should utilize the ability to click on a plot or select a region in some way.</li>
    <li> You should include some type of math type (maybe an equation or just a special symbol you need to use mathJax for).</li>
    <li> You should include a link to something and some other formatted text.</li>
    <li> Modeling
    <ul>
      <li>(At least) two supervised learning models (feel free to branch out to things we didn’t discuss if you’d like)</li>
      <li> You should give the user some functionality for choosing model settings (variables used, number of trees, etc.) and for changing relevant output</li>
      <li> You should give the user a way to use the model for prediction (they should be able to select the values of the predictors)</li>
    </ul>
    </li>
</ul>

"))),
                   
                   tabPanel("Introduction to Data",
                            mainPanel(
                              HTML("
<h3>Effect of Surface and Vision on Balance (copy from <a href='http://www.statsci.org/data/oz/ctsib.html'>www.statsci.org</a>)</h3>
<p>Keywords: repeated measures, ordinal regression</p>
<hr><h3>Description</h3>
<p>The balance of subjects were observed for two different surfaces and for restricted and unrestricted vision. Balance was assessed qualitatively on an ordinal 4-point scale based on observation by the experimenter. Subjects were expected to be better balanced (show less sway) when standing on the normal surface than on foam, and when their eyes were open rather than closed or when their vision was restricted by a dome. </p>
<p>Equal numbers of male and female subjects were chosen.  For both males and females, ten older (more than 24 years old) and ten younger subjects were selected.</p>
<p>The data is available in two formats. The is in univariate or &quot;strung out form&quot; which is suitable for entry to Minitab or S-Plus and to most mixed model programs. The second is in repeated measures format which is suitable for SPSS and for most special purpose repeated measures programs.</p>
<p>Univariate format:</p>
<table cellpadding='0' cellspacing='0'>
    <tr><td widht='20'><td colspan='3'><hr></td></tr>
    <tr><td width='20'><td align='right'>Variable <td width='20'><td>Description</tr>
    <tr><td widht='20'><td colspan='3'><hr></td></tr>
    <tr><td width='20'><td align='right'>Subject<td width='20'><td>1 to 40</tr>
    <tr><td width='20'><td align='right'>Sex<td width='20'><td>male or female</tr>
    <tr><td width='20'><td align='right'>Age<td width='20'><td>Age of subject in years</tr>
    <tr><td width='20'><td align='right'>Height<td width='20'><td>Height in cm</tr>
    <tr><td width='20'><td align='right'>Weight<td width='20'><td>Weight in kg</tr>
    <tr><td width='20'><td align='right'>Surface<td width='20'><td>normal or foam</tr>
    <tr><td width='20'><td align='right'>Vision<td width='20'><td>eyes open, eyes closed, or closed dome</tr>
    <tr><td width='20'><td align='right'>CTSIB<td width='20'><td>Qualitive measure of balance, 1 (stable) - 4 (unstable)
    <tr><td widht='20'><td colspan='3'><hr></td></tr>
</table>
<p>Repeated measures format:</p>
      <table cellpadding='0' cellspacing='0'>
      <tr><td widht='20'><td colspan='3'><hr></td></tr>
      <tr><td width='20'><td align='right'>Variable <td width='20'><td>Description</tr>
      <tr><td widht='20'><td colspan='3'><hr></td></tr>
      <tr><td width='20'><td align='right'>Subject<td width='20'><td>1 to 40</tr>
      <tr><td width='20'><td align='right'>Sex<td width='20'><td>male or female</tr>
      <tr><td width='20'><td align='right'>Age<td width='20'><td>Age of subject in years</tr>
      <tr><td width='20'><td align='right'>Height<td width='20'><td>Height in cm</tr>
      <tr><td width='20'><td align='right'>Weight<td width='20'><td>Weight in kg</tr>
      <tr>
      <td width='20'><td align='right'>NO1<td width='20'><td>Balance measure on
      normal surface with eyes open, first replicate
      </tr>
      <tr>
      <td width='20'><td align='right'>NO2<td width='20'><td>as above, second
      replicate
      </tr>
      <tr>
      <td width='20'><td align='right'>NC1<td width='20'><td>Balance measure on
      normal surface with eyes closed, first replicate
      </tr>
      <tr>
      <td width='20'><td align='right'>NC2<td width='20'><td>as above, second
      replicate
      </tr>
      <tr>
      <td width='20'><td align='right'>ND1<td width='20'><td>Balance measure on
      normal surface with dome, first replicate
      </tr>
      <tr>
      <td width='20'><td align='right'>ND2<td width='20'><td>as above, second
      replicate
      </tr>
      <tr>
      <td width='20'><td align='right'>FO1<td width='20'><td>Balance measure on foam
      surface with eyes open, first replicate
      </tr>
      <tr><td width='20'><td align='right'>FO2<td width='20'><td>as above, second
      replicate</tr>
      <tr>
      <td width='20'><td align='right'>FC1<td width='20'><td>Balance measure on foam
      surface with eyes closed, first replicate
      </tr>
      <tr><td width='20'><td align='right'>FC2<td width='20'><td>as above, second
      replicate</tr>
      <tr>
      <td width='20'><td align='right'>FD1<td width='20'><td>Balance measure on foam
      surface with dome, first replicate
      </tr>
      <tr><td width='20'><td align='right'>FD2<td width='20'><td>as above, second
      replicate
      <tr><td widht='20'><td colspan='3'><hr></td></tr>
      </table>
      
      <h3>Download</h3>
      
      <ul>
      <li><a href='http://www.statsci.org/data/oz/ctsibuni.txt'>Data file</a> (tab-delimited text, univariate 
      format)</li>
      <li><a href='http://www.statsci.org/data/oz/ctsibrm.txt'>Data file</a> (repeated measures format)</li>
      </ul>
      
      <h3>Source</h3>
      
      <table cellspacing='0' cellpadding='0'>
      <tr>
      <td>Steele, R. (1998). Honours Thesis, Department of Physiotherapy, University of Queensland. </td>
      </table>
      
      <h3>Analysis</h3>
      
      <p>The data is best analysed using ordinal methods, but normal analysis of variance gives a good approximation. We see strong effects for Surface and Vision, and an interesting interaction between Sex and Vision. If ordinal regression is used, the two 4's in the response will need to be recoded to 3's.</p>
                      ")
                            )),
                   # Application title
                   tabPanel("Exploration of the ctsib data",
                            # Sidebar with options for the data set
                            sidebarLayout(
                              sidebarPanel(
                                h3("Select the Vision:"),
                                selectizeInput("Vision", "Vision", selected = "open", choices = levels(as.factor(ctsib$Vision))),
                                br(),
                                h3("Select the Surface:"),
                                selectizeInput("Surface", "Surface", selected = "norm", choices = levels(as.factor(ctsib$Surface))),
                                br(),
                                h3("Select the Gender:"),
                                selectizeInput("Sex", "Sex", selected = "all", choices = c(levels(as.factor(ctsib$Sex )),"all")),
                                br(),
                                checkboxInput("hideplot", "Hide the plot"),
                                sliderInput("size", "Size of Points on Graph", min = 1, max = 10, value = 3, step = 1),
                                conditionalPanel(condition = "input.Sex == 'all'",
                                                 checkboxInput("conservation", h5("Color Code based on gender (only shows when all is selected)", style = "color:red;"))
                                )
                              ),
                              
                              # Show output
                              mainPanel(
                                helpText(a("Click Here to see my datasource",href="http://www.statsci.org/data/oz/ctsibuni.txt")
                                ),
                                conditionalPanel(condition = "input.hideplot == false",
                                                 downloadButton('downloadPlot', 'Download Plot'),
                                                 
                                                 plotOutput("ctsibPlot", height = 300,
                                                            # Equivalent to: click = clickOpts(id = "plot_click")
                                                            click = "plot1_click",
                                                            brush = brushOpts(id = "plot1_brush")
                                                 )),
                                
                                fluidPage(
                                  fluidRow(
                                    column(width = 8,
                                           h4("Points near click"),
                                           verbatimTextOutput("click_info")
                                    ),
                                    column(width = 8,
                                           h4("Brushed points"),
                                           verbatimTextOutput("brush_info")
                                    )
                                  )
                                ),
                                downloadButton("downloadData", "Download"),
                                tableOutput("table")
                              )
                            )
                   ),
                   
                   # Using KNN to analyze the data
                   tabPanel("Analysis with KNN/Logistic regression",
                            # Sidebar with options for the data set
                            sidebarLayout(
                              sidebarPanel(
                                h3("Select the supervised learning method:"),
                                selectizeInput("SupervisedMethod", "SupervisedMethod", selected = "knn", choices = c('knn','Logistic regression')),
                                h3("Training data percentage:"),
                                sliderInput("TrainingPercent", "Percent of data used for training", min = 30, max = 80, value = 60, step = 1),
                                sliderInput("NeighborCount", "Number of neighbors used", min = 1, max = 20, value = 10, step = 1),
                                selectInput("SelectedColumns","Select Predictors", choices =c("Sex","Age","Height","Weight","Surface","Vision","BMI","PI" ),multiple = TRUE,selected = c("Sex","Age","Height","Weight","Surface","Vision","BMI","PI"))
                              ),
                              
                              # Show output
                              mainPanel(
                                helpText("The first column is the predicted CTSIB, while the Column name is observed value"),
                                uiOutput("knn_output"),
                                uiOutput("misclassifiedPct")
                              )
                            )
                   ),
                   
                   # Using KNN to analyze the data
                   tabPanel("Analysis with unsupervised learning",
                            # Sidebar with options for the data set
                            sidebarLayout(
                              sidebarPanel(
                                selectizeInput("NumberOfPcs", "Principal Component to display", selected = "1", choices = as.factor(c(1:6))),
                                #sliderInput("NumberOfPcs", "Number of PCs", min = 1, max = 6, value = 1, step = 1),
                                selectInput("PcaSelectedColumns","Select Predictors", choices =c("Sex","Age","Height","Weight","Surface","Vision","BMI","PI" ),multiple = TRUE,selected = c("Sex","Age","Height","Weight","Surface","Vision","BMI","PI")),
                                checkboxInput("plotEllipse", "Show the ellipse", value = TRUE),
                                checkboxInput("plotCircle", "Show the circle", value = TRUE),
                                
                                helpText("Based on the WIKI page for PCA, in order to maximize variance, the first weight vector  thus has to satisfy"),
                                withMathJax(helpText("$$\\LARGE w_{(1)}=\\arg\\max_{||{w}||=1}\\{\\sum_i{(t_i)_{(i)}^2}\\}=\\arg\\max_{||{w}||=1}\\{\\sum_i{(x_{(i)}*w)^2}\\}$$")),
                                withMathJax(helpText("Since \\(\\LARGE w_{(1)}\\) has been defined to be a unit vector, it equivalently also satisfies")),
                                withMathJax(helpText("$$\\LARGE w_{(1)}=\\arg\\max_{\\left||{w}\\right||=1}{\\{\\frac{w^TX^TXw}{w^Tw}}\\}$$")),
                                helpText("Further components can be defined by following formula"),
                                withMathJax(helpText("$$\\LARGE \\widehat{X}_k=X-\\sum_{\\delta=1}^{k-1}Xw_{{(\\delta)}}w_{{(\\delta)}}^T$$"))
                              ),
                              # Show output
                              mainPanel(
                                h3(textOutput('caption')),
                                tabsetPanel(
                                  tabPanel("Component Plot",plotOutput("componentplot",height = 280*2, width = 250*2)),
                                  tabPanel("Scree Plot",plotOutput("screeplot", height = "300px")),
                                  tabPanel("Data sample",tableOutput("filetable"))
                                )
                                
                              )
                            )
                   )
                   
))