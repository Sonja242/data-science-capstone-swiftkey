# Sonja Next Word | Author: Sonja Sahebzad | Sonja Projects
library(shiny)
source("predictor.R", local = TRUE)
model <- prepare_predictor(readRDS("model.rds"))
metrics <- jsonlite::fromJSON("metrics.json")
runtime <- jsonlite::fromJSON("runtime-metrics.json")
pct <- function(x) sprintf("%.1f%%", 100*x)
m <- metrics$overall

ui <- fluidPage(
  title = "Sonja Next Word",
  tags$head(tags$link(rel="stylesheet",type="text/css",href="styles.css"),
    tags$meta(name="viewport",content="width=device-width, initial-scale=1"),
    tags$script(src="input.js")),
  tags$div(class="shell",
    tags$header(class="masthead", tags$a(class="brand", href="#", "SONJA PROJECTS"),
      tags$span(class="edition", "DATA SCIENCE CAPSTONE")),
    tags$div(class="hero", tags$p(class="eyebrow","A SMALL ASSIST FOR YOUR NEXT SENTENCE"),
      tags$h1("Find your next word."),
      tags$p(class="intro", "Type an English phrase. Get one suggestion, plus two alternatives to keep your writing moving.")),
    tabsetPanel(id="view",type="tabs",
      tabPanel("Write",value="write",
        tags$div(class="writing-grid",
          tags$section(class="editor-panel",tags$h2("Start with a phrase"),
            textAreaInput("phrase","Your English phrase",value="",rows=4,
              placeholder="The weather today is",width="100%"),
            tags$div(class="input-meta",tags$span("Up to 500 characters. Add a space after a complete word."),textOutput("counter",inline=TRUE)),
            checkboxInput("automatic","Update suggestions while writing",value=TRUE),
            tags$div(class="actions",actionButton("predict","Predict next word",class="btn-primary"),actionButton("clear","Clear",class="btn-quiet")),
            tags$p(class="keyboard-hint","Automatic updates follow a space or punctuation and a short pause. Without a space, press Predict or Ctrl + Enter (Command + Enter on Mac)."),
            tags$p(class="example-label","Try a starting point"),
            tags$div(class="examples",actionButton("example1","The weather today is"),
              actionButton("example2","I am looking forward to"),actionButton("example3","She put the book on")),
            tags$p(class="privacy-note","Text is processed by the Shiny server for this session. This app does not save your phrases or send them to an external AI service.")),
          tags$section(class="prediction-panel",`aria-live`="polite",`aria-atomic`="true",
            uiOutput("prediction"))),
        tags$div(class="under-note", tags$strong("A suggestion, not a certainty. "),
          "The model learns common word patterns. It can miss meaning, new names and unusual phrasing. You decide what fits.")),
      tabPanel("Results",value="results",
        tags$section(class="content-panel",tags$p(class="eyebrow","MEASURED ON THE MODEL RUNNING HERE"),
          tags$h2("Useful suggestions. Transparent evidence."),
          tags$p("The frozen app model was tested on 900 previously unused text examples: 300 each from blogs, news and Twitter. The actual next word was hidden before prediction."),
          tags$div(class="metric-grid",
            tags$div(class="metric",tags$span("First suggestion correct"),tags$strong(pct(m$top1)),tags$small(sprintf("%d of 900; 95%% interval %s to %s",m$top1_count,pct(m$top1_low),pct(m$top1_high)))),
            tags$div(class="metric",tags$span("Correct within three"),tags$strong(pct(m$top3)),tags$small(sprintf("%d of 900; 95%% interval %s to %s",m$top3_count,pct(m$top3_low),pct(m$top3_high)))),
            tags$div(class="metric",tags$span("Local computation, median"),tags$strong(sprintf("%.2f ms",runtime$median_ms)),tags$small(sprintf("95th percentile %.2f ms; 3,000 local calls; network time excluded",runtime$p95_ms)))),
          tags$h3("Performance by source"),
          plotOutput("source_plot",height="250px"),
          tableOutput("source_table"),
          tags$p(class="fine-print","Intervals describe sampling uncertainty, not confidence in an individual suggestion. Equal source weighting may differ from real usage. The app displays no calibrated confidence percentage."),
          tags$h3("The deployment trade-off"),
          tags$p(sprintf("The %s configuration was selected on 600 separate development cases from three compact candidates. Its %s-word vocabulary and up-to-%d-gram context keep the trained model to %.1f MiB on disk (%.1f MiB prepared for fast lookup).",
            tolower(metrics$selected),format(metrics$vocabulary,big.mark=","),metrics$max_order,metrics$model_mib,runtime$prepared_memory_mib)),
          tags$p("These are unrestricted next-word results, not multiple-choice quiz scores. Local timing uses five randomized rounds on the same 600 development inputs. Exact indexed lookup preserved all outputs on 5,510 regression phrases. Timing measures computation after loading; the 150 ms automatic typing delay, network and display add to total waiting. A first visit may take longer while the hosting service wakes the app."))),
      tabPanel("How to use",value="guide",
        tags$section(class="content-panel guide",tags$h2("Three steps to keep writing"),
          tags$ol(tags$li("Enter an English phrase, or choose a starting point."),
            tags$li("Finish a word and add a space. Suggestions update after a short pause. Alternatively, select Predict next word."),
            tags$li("Select a suggestion to add it, or keep typing your own words. The next suggestions update automatically.")),
          tags$h3("How the prediction works"),
          tags$p(sprintf("The model was trained on %s deduplicated English lines from the official SwiftKey corpus. It combines patterns of up to %d words, using at most the last %d words of your phrase. Longer patterns help when they have support; shorter patterns provide a fallback when they do not.",format(metrics$training_lines,big.mark=","),metrics$max_order,metrics$max_order-1L)),
          tags$p("Kneser-Ney smoothing shares probability with shorter patterns. Pruning retains a compact set of useful continuations. Sparse scoring produces the same top suggestions as scoring every word in this compact model, while avoiding unnecessary computation."),
          tags$h3("What to expect"),
          tags$p("Suggestions are lowercase English words. Capitalization, numbers, URLs and punctuation are normalized. Unfamiliar contexts fall back to frequent words. This is next-word completion, not spelling correction or sentence-level reasoning."),
          tags$p("A full sentence is accepted, but only the most recent four normalized words can affect the prediction. The context shown under the suggestions identifies the longest matched pattern. Informal spellings, such as mornin, can occur because the training corpus includes informal writing."),
          tags$p("An explicit output blocklist excludes some common profanities. It is not a comprehensive content filter. The short context can overlook the meaning of the full sentence."),
          tags$h3("Evidence and reproducibility"),
          tags$p("The model weights and vocabulary remain fixed during use. There is no online learning from your input. Development selection and the independent final test are recorded separately in the project source."),
          tags$ul(
            tags$li(tags$a(href="https://github.com/Sonja242/data-science-capstone-swiftkey/tree/main/final_project",target="_blank",rel="noopener","Project source and reproduction instructions")),
            tags$li(tags$a(href="https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip","Official Coursera SwiftKey corpus")),
            tags$li(tags$a(href="https://aclanthology.org/P96-1041/",target="_blank",rel="noopener","Chen and Goodman (1996): smoothing techniques")),
            tags$li(tags$a(href="https://shiny.posit.co/r/getstarted/shiny-basics/lesson1/",target="_blank",rel="noopener","Posit: Shiny for R"))))
      )
    ),
    tags$footer(tags$span("Sonja Next Word"),tags$span("Author: Sonja Sahebzad"),tags$span("Sonja Projects · 2026"))
  )
)

server <- function(input, output, session) {
  result <- reactiveVal(NULL)
  error <- reactiveVal(NULL)
  last_phrase <- reactiveVal("")
  output$counter <- renderText(sprintf("%s / 500",nchar(input$phrase %||% "")))
  observeEvent(input$phrase,{result(NULL);error(NULL)},ignoreInit=TRUE,priority=10)
  settled_phrase <- debounce(reactive(input$phrase %||% ""), millis=150)
  predict_for <- function(phrase, explicit=FALSE) {
    if (nchar(phrase)>500L) {result(NULL);error("Please shorten the phrase to 500 characters or fewer.");return()}
    if (!nzchar(normalize_phrase(phrase))) {
      result(NULL)
      error(if(explicit) "Enter an English word or phrase, then select Predict next word." else NULL)
      return()
    }
    if (identical(phrase,last_phrase()) && !is.null(result())) return()
    start <- as.numeric(Sys.time())
    p <- predict_word(model,phrase)
    p$elapsed <- (as.numeric(Sys.time())-start)*1000
    last_phrase(phrase);error(NULL);result(p)
  }
  observeEvent(input$predict,{
    predict_for(input$phrase %||% "",explicit=TRUE)
  })
  observeEvent(list(settled_phrase(),input$automatic),{
    phrase <- settled_phrase()
    # A pending timer must never replace a newer input or a manual prediction.
    if (!isTRUE(input$automatic) || !identical(phrase,input$phrase %||% "")) return()
    if (nchar(phrase)>500L || grepl("[[:space:].!?;:,]$",phrase)) predict_for(phrase)
  })
  observeEvent(input$clear,{updateTextAreaInput(session,"phrase",value="");result(NULL);error(NULL)})
  examples <- c("The weather today is","I am looking forward to","She put the book on")
  for (i in seq_along(examples)) local({j <- i
    observeEvent(input[[paste0("example",j)]],{updateTextAreaInput(session,"phrase",value=paste0(examples[j]," "));result(NULL);error(NULL)})
  })
  for (i in 1:3) local({j <- i
    observeEvent(input[[paste0("choose",j)]],{
      p <- result();req(p,!is.null(p$words[j]),identical(input$phrase,last_phrase()))
      new_text <- paste0(trimws(last_phrase())," ",p$words[j]," ")
      if(nchar(new_text)>500L){error("Adding this word would exceed 500 characters.");return()}
      updateTextAreaInput(session,"phrase",value=new_text);result(NULL);error(NULL)
    })
  })
  output$prediction <- renderUI({
    p <- result()
    if (!is.null(error())) return(tagList(tags$p(class="eyebrow","ONE MORE STEP"),tags$h2("Ready when you are."),tags$p(class="input-error",error())))
    if (is.null(p)) return(tagList(tags$p(class="eyebrow","YOUR NEXT WORD"),
      tags$div(class="empty-mark", "Aa"),tags$h2("A little help with what comes next."),
      tags$p(if(isTRUE(input$automatic)) "Finish a word and add a space, or select Predict next word. Suggestions refresh as you continue." else "Enter a phrase and select Predict next word. Your suggestions will appear here.")))
    tags$div(`data-prediction-phrase`=last_phrase(),tags$p(class="eyebrow","FIRST SUGGESTION"),
      actionButton("choose1",p$words[1],class="word-primary",title="Add this word to your phrase"),
      tags$p(class="choose-hint","Select a word to add it to your phrase."),
      tags$p(class="example-label","Two alternatives"),
      tags$div(class="alternatives",actionButton("choose2",p$words[2]),actionButton("choose3",p$words[3])),
      tags$div(class="evidence",tags$strong(if(p$order==1L) "Frequent-word fallback" else sprintf("Pattern with %d recent word%s",p$context_words,if(p$context_words==1L)"" else "s")),
        tags$p(if(p$order==1L)"This context was not retained in the compact model." else paste("Context:",if(nzchar(p$context))p$context else "start of text")),
        tags$p(class="context-note",sprintf("%d input word%s. Predictions use at most %d recent words.",p$input_words,if(p$input_words==1L)"" else "s",model$maximum_order-1L)),
        tags$small(sprintf("%.0f ms to compute on this server. Network time excluded.",p$elapsed))))
  })
  output$source_plot <- renderPlot({
    d <- metrics$by_source
    par(mar=c(4,4,1,1),family="sans",fg="#173b59",col.axis="#173b59")
    bp <- barplot(rbind(100*d$top1,100*d$top3),beside=TRUE,names.arg=d$source,
      col=c("#173f66","#75aacf"),border=NA,ylim=c(0,max(d$top3_high*100)+8),
      ylab="Correct predictions (%)",legend.text=c("First suggestion","Within three"),
      args.legend=list(x="topleft",bty="n",horiz=TRUE,cex=.85))
    arrows(bp[1,],100*d$top1_low,bp[1,],100*d$top1_high,angle=90,code=3,length=.04)
    arrows(bp[2,],100*d$top3_low,bp[2,],100*d$top3_high,angle=90,code=3,length=.04)
  },res=110,alt="First-suggestion and top-three accuracy by text source. Exact values appear in the table below.")
  output$source_table <- renderTable({
    d <- metrics$by_source
    data.frame(Source=d$source,Cases=d$cases,`First suggestion`=pct(d$top1),
      `Within three`=pct(d$top3),check.names=FALSE)
  },striped=TRUE,spacing="s")
}
`%||%` <- function(a,b) if (is.null(a)) b else a
shinyApp(ui,server)
