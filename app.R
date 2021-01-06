
#  #1 - MARKET BASKET ANALYZER

# LIBRARIES ----
library(shiny)
library(shinythemes)

# Tables
library(DT)
library(knitr)
library(kableExtra)

# Core
library(tidyverse)
library(tidyquant)

# Preprocessing
library(recipes)

# Python
library(reticulate)

# PYTHON SETUP ----

# Replace this with your conda environment containking sklearn, pandas, & numpy
use_condaenv("py3.8", required = TRUE)
source_python("py/logistic_reg.py")

# DATA SETUP ----
ecommerce_raw_tbl  <- read_csv("data/ecommerce_data.csv")
invoice_selections <- ecommerce_raw_tbl %>%
    filter(!InvoiceNo %>% str_detect("^C")) %>%
    distinct(InvoiceNo) %>% 
    slice(1:30) %>%
    pull()

sample_data <- ecommerce_raw_tbl %>% filter(InvoiceNo %in% invoice_selections)

customer_order_history_tbl <- read_rds("preprocessing/customer_habits_joined_tbl.rds") %>%
    select(CustomerID, count)

product_clusters_tbl <- read_rds("preprocessing/product_clusters_tbl.rds") 

recipe_spec <- read_rds("preprocessing/recipe_spec_customer_prediction.rds")

cluster_morphology_tbl <- read_rds("preprocessing/cluster_morphology_tbl.rds")

cluster_morphology_ggplot <- read_rds("preprocessing/cluster_morphology_ggplot.rds")

# INFO CARD ----
info_card <- function(title, value, sub_value = NULL,
                      main_icon = "chart-line", sub_icon = "arrow-up",
                      bg_color = "default", text_color = "default", 
                      sub_text_color = "success") {
    
    div(
        class = "panel panel-default",
        style = "padding: 0px;",
        div(
            class = str_glue("panel-body bg-{bg_color} text-{text_color}"),
            p(class = "pull-right", icon(class = "fa-4x", main_icon)),
            h4(title),
            h5(value),
            p(
                class = str_glue("text-{sub_text_color}"),
                icon(sub_icon),
                tags$small(sub_value)
            )
        )
    )
    
}

# UI ----
ui <- navbarPage(
    title = "E-Commerce App",
    collapsible = TRUE,
    position    = "static-top", 
    inverse     = TRUE, 
    theme       = shinytheme("paper"),
    
    tabPanel(
        title = "Market Basket Analyzer",
        sidebarLayout(
            sidebarPanel = sidebarPanel(
                width = 3,
                h3("Market Basket Analyzer"),
                HTML("<p>Product recommendations are a key component of maximizing revenue.
                  Select an invoice, and perform <strong>Market Basket Analysis</strong> and 
                     recommendation using <code>Python Scikit Learn</code>.</p>"),
                hr(),
                shiny::selectInput(
                    inputId  = "invoice_selection", 
                    label    = "Analyze an Invoice",
                    choices  = invoice_selections,
                    selected = invoice_selections[1]
                        
                ),
                shiny::actionButton(inputId = "submit", "Submit", class = "btn-primary"),
                hr(),
                # uiOutput("text"),
                br(),
                h5("Customer Order"),
                htmlOutput("basket_small")
                
            ),
            mainPanel = mainPanel(
                width = 9,
                # * VALUE BOXES ----
                uiOutput("value_boxes"),
                div(
                    class = "row",
                    div(
                        class = "col-sm-12 panel",
                        div(class = "panel-heading", h5("Customer Segment Morphology")),
                        div(
                            class = "panel-body",
                            plotOutput("ggplot", height = "275px"),
                            # verbatimTextOutput(outputId = "print")
                        )
                    )
                ),
                div(
                    class = "row",
                    tabsetPanel(
                        tabPanel(
                            title = "Recommended Products",
                            div(
                                class = "col-sm-12 panel",
                                div(class = "panel-heading", h5("Recommended Products")),
                                div(
                                    class = "panel-body",
                                    dataTableOutput("recommendation")
                                )
                            )
                        ),
                        tabPanel(
                            title = "Full Customer Basket",
                            div(
                                class = "col-sm-12 panel",
                                div(class = "panel-heading", h5("Full Customer Market Basket (Invoice)")),
                                div(
                                    class = "panel-body",
                                    dataTableOutput("basket")
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

# SERVER ---- 
server <- function(session, input, output) {
    rv <- reactiveValues()
    
    # DATA PREPARAION ----
    observeEvent(input$submit, {
        
        rv$invoice_tbl <- ecommerce_raw_tbl %>%
            filter(InvoiceNo %in% input$invoice_selection) %>%
            mutate(PriceExt = UnitPrice * Quantity) %>%
            left_join(product_clusters_tbl %>% select(StockCode, cluster))
        
        # Count (Invoice History)
        customer_id <- rv$invoice_tbl %>%
            distinct(CustomerID) %>%
            pull()
        
        rv$count <- customer_order_history_tbl %>%
            filter(CustomerID == customer_id) %>%
            slice(1) %>%
            pull(count)
        
        
        # Mean (Total Invoice Value)
        rv$mean <- rv$invoice_tbl %>%
            pull(PriceExt) %>%
            sum(na.rm = TRUE)
        
        # Category Proportions (Product Clusters)
        empty_tbl <- c(str_c("cat_", 0:5), "cat_NA") %>% 
            purrr::map_dfc(setNames, object = list(numeric()))
        
        rv$product_props_by_cat <- rv$invoice_tbl %>%
            select(CustomerID, PriceExt, cluster) %>%
            group_by(CustomerID, cluster) %>%
            summarize(PriceExt = SUM(PriceExt)) %>%
            mutate(prop = PriceExt / SUM(PriceExt)) %>%
            ungroup() %>%
            select(-PriceExt) %>%
            pivot_wider(names_from   = cluster, 
                        values_from  = prop, 
                        values_fill  = list(prop = 0), 
                        names_prefix = "cat_") %>%
            select(-CustomerID) %>%
            bind_rows(empty_tbl)
        
        # Fill missing categories with zero    
        rv$product_props_by_cat[is.na(rv$product_props_by_cat)] <- 0
        
        # Make preprocessing table
        rv$data_bind <- tibble(
            count = rv$count,
            mean  = rv$mean
        ) %>%
            bind_cols(rv$product_props_by_cat) 
        
        # Apply recipe & reorder to match column structure for previously trainedLogistic Regression Model
        rv$data_prep <- bake(recipe_spec, new_data = rv$data_bind) %>%
            select(count, mean, cat_2, cat_0, cat_1, cat_3, cat_NA, cat_4)
        
        # Make prediction
        rv$prediction <- lr_predict(as.matrix(rv$data_prep)) %>% as.character()
        
        # Product Recommendation
        product_cat <- cluster_morphology_tbl %>%
            mutate(cluster = as.character(cluster) %>% as.numeric()) %>%
            filter(cluster == rv$prediction) %>%
            select(cluster, contains("cat_")) %>%
            pivot_longer(-cluster) %>%
            arrange(desc(value)) %>%
            slice(1) %>%
            separate(name, into = c("unnecessary", "product_cat")) %>%
            mutate(product_cat = as.numeric(product_cat)) %>%
            pull(product_cat) %>%
            as.character()
        
        if (is.na(product_cat)) product_cat <- "5"
        
        rv$recommendation_tbl <- ecommerce_raw_tbl %>%
            select(StockCode, Quantity) %>%
            group_by(StockCode) %>%
            summarize(Quantity = SUM(Quantity)) %>%
            ungroup() %>%
            left_join(product_clusters_tbl) %>%
            mutate(product_category = as.character(cluster)) %>%
            mutate(product_category = ifelse(is.na(product_category), "5", product_category)) %>%
            filter(product_category == product_cat) %>%
            arrange(desc(Quantity)) %>%
            slice(1:3) %>%
            select(StockCode, mode_description, median_unit_price, product_category)
        
     
        
    }, ignoreNULL = FALSE)
    
    # Debugging ----
    output$print <- renderPrint({
        list(
            invoice_tbl    = rv$invoice_tbl,
            # data_prep    = rv$data_prep,
            prediction     = rv$prediction,
            recommendation = rv$recommendation_tbl
        )
    })
    
    # VALUEBOX OUTPUT ----
    output$value_boxes <- renderUI({
        
        order_count   <- rv$count
        invoice_value <- rv$invoice_tbl %>% pull(PriceExt) %>% SUM()
        prediction    <- rv$prediction
        
        tagList(
            column(
                width = 4,
                info_card(
                    title = HTML("<span style='color:white;'>Order History</span>"), 
                    value = HTML(str_glue("<span class='label label-info'>{order_count}</span>")), 
                    sub_value = ifelse(order_count > 4, "Above Average", "Below Average"), 
                    sub_icon  = ifelse(order_count > 4, "arrow-up", "arrow-down"),
                    bg_color  = "primary", 
                    sub_text_color = ifelse(order_count > 4, "default", "danger"), 
                    main_icon = "credit-card"
                )
            ),
            column(
                width = 4,
                info_card(
                    title = HTML("<span style='color:white;'>Invoice Value</span>"), 
                    value = HTML(str_glue("<span class='label label-info'>{scales::dollar(invoice_value)}</span>")), 
                    sub_value = ifelse(invoice_value > 250, "Above Average", "Below Average"), 
                    sub_icon  = ifelse(invoice_value > 250, "arrow-up", "arrow-down"),
                    bg_color  = ifelse(invoice_value > 250, "primary", "danger"), 
                    sub_text_color = ifelse(invoice_value > 250, "default", "danger"),
                    main_icon = "money-bill-wave"
                )
            ),
            column(
                width = 4,
                info_card(
                    title = HTML("<span style='color:white;'>Customer Segment</span>"), 
                    value = HTML(str_glue("<span class='label label-info'>{prediction}</span>")), 
                    sub_value = case_when(
                        prediction == "0" ~ "Big Spender! Offer discounts for bulk.",
                        prediction == "1" ~ "Promote Product Categories 1 & 2",
                        prediction == "2" ~ "Difficult to Classify",
                        prediction == "3" ~ "Promote Product Categories 0 & 2",
                        prediction == "4" ~ "Promote Product Category 2",
                        prediction == "5" ~ "Promote Product Categories: 1, 3, 4"
                    ), 
                    sub_icon  = NULL,
                    bg_color  = "primary", 
                    sub_text_color = "default",
                    main_icon = "users"
                )
            )
        )
    })
    
    # KNITR TABLE OUTPUT ----
    output$basket_small <- renderText({
        rv$invoice_tbl %>%
            rename(`Product Category` = cluster) %>%
            select(StockCode, Description) %>%
            kable() %>%
            kable_styling(
                font_size = 10,
                bootstrap_options = c("striped", "hover", "condensed")
            )
    })
    
    # DATA TABLE FULL INVOICE ----
    output$basket <- renderDataTable({
        rv$invoice_tbl %>%
            rename(`ProductCategory` = cluster) %>%
            select(InvoiceDate, StockCode, ProductCategory, Description, 
                   CustomerID, Country, Quantity, UnitPrice, PriceExt) %>%
            datatable(options = list(dom = "t")) %>%
            formatCurrency(columns = c("UnitPrice", "PriceExt"))
    })
    
    # DATA TABLE RECOMMENDATION ----
    output$recommendation <- renderDataTable({
        rv$recommendation_tbl %>%
            set_names(c("Stock Code", "Description", "Unit Price (Median)", "Product Category")) %>%
            datatable(options = list(dom = "t")) %>%
            formatCurrency(columns = "Unit Price (Median)")
    })
    
    # # TEXT PREDICTION ----
    # output$text <- renderUI({
    #     div(
    #         h5("Customer Segment Prediction: ", span(rv$prediction, class="label label-warning"))
    #     )
    # })
    
    # GGPLOT SEGMENT MORPHOLOGY ----
    output$ggplot <- renderPlot({
        cluster_morphology_ggplot +
            facet_wrap(~ cluster, nrow = 1) +
            theme(
                strip.text = element_text(color = "white"),
                plot.margin = unit(c(0,0,0,0),"mm"), 
                text = element_text(size = 16)
            ) +
            labs(title = "")
    })
    
    
}

# Run the application 
shinyApp(ui = ui, server = server)