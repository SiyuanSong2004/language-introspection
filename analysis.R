library(lme4) 
library(tidyverse)
library(forcats)
library(psych)
library(patchwork)
library(plotrix)
library(lmerTest)
library(ggtext)
library(showtext)
library(viridis)
library(texreg)
showtext_auto()
font_add_google("EB Garamond", "garamond")
####################global settings##########################
sim_colors<-c('#d76f56', '#be76c3', '#718ed6', '#92c0da', '#757575')
olmo_colors<-c('#d76f56', '#be76c3', '#92c0da')

####################read and process data################################
d.expr1 = read_csv("alldata/expr1.csv") %>%
  mutate(seed = case_when(grepl("ingredient1", model) ~ 1,
                          grepl("ingredient2", model) ~ 2,
                          grepl("ingredient3", model) ~ 3,
                          
                          TRUE ~ 0)) 

d.expr2 = read_csv("alldata/expr2.csv") %>%
  mutate(seed = case_when(grepl("ingredient1", model) ~ 1,
                          grepl("ingredient2", model) ~ 2,
                          grepl("ingredient3", model) ~ 3,
                          TRUE ~ 0)) 
expr2.datasets <- c('wikipedia','news','nonsense','randomseq')

# change model names
d.expr1 <- d.expr1 %>%
  mutate(model = case_when(
    model == "olmo-2-1124-7b-stage2-ingredient1-step11931-tokens50B" ~ "OLMo 7B-seed1",
    model == "olmo-2-1124-7b-stage2-ingredient2-step11931-tokens50B" ~ "OLMo 7B-seed2",
    model == "olmo-2-1124-7b-stage2-ingredient3-step11931-tokens50B" ~ "OLMo 7B-seed3",
    model == "olmo-2-1124-13b-stage2-ingredient1-step11931-tokens100B" ~ "OLMo 13B-seed1",
    model == "olmo-2-1124-13b-stage2-ingredient2-step11931-tokens100B" ~ "OLMo 13B-seed2",
    model == "olmo-2-1124-13b-stage2-ingredient3-step11931-tokens100B" ~ "OLMo 13B-seed3",
    model == "olmo-2-1124-7b" ~ "OLMo 7B",
    model == "olmo-2-1124-13b" ~ "OLMo 13B",
    
    model == "qwen2.5-1.5b" ~ "Qwen 1.5B",
    model == "qwen2.5-1.5b-instruct" ~ "Qwen 1.5B IT",
    model == "qwen2.5-7b" ~ "Qwen 7B",
    model == "qwen2.5-7b-instruct" ~ "Qwen 7B IT",
    model == "qwen2.5-72b" ~ "Qwen 72B",
    model == "qwen2.5-72b-instruct" ~ "Qwen 72B IT",
    
    model == "llama-3.1-8b" ~ "Llama3.1 8B",
    model == "llama-3.1-8b-instruct" ~ "Llama3.1 8B IT",
    model == "llama-3.1-70b" ~ "Llama3.1 70B",
    model == "llama-3.1-70b-instruct" ~ "Llama3.1 70B IT",
    model == "llama-3.1-405b" ~ "Llama3.1 405B",
    model == "llama-3.3-70b-instruct" ~ "Llama3.3 70B IT",
    
    model == "mistral-large-instruct-2411" ~ "Mistral 123B IT",
    
    TRUE ~ model
  ))


d.expr2 <- d.expr2 %>%
  mutate(model = case_when(
    model == "olmo-2-1124-7b-stage2-ingredient1-step11931-tokens50B" ~ "OLMo 7B-seed1",
    model == "olmo-2-1124-7b-stage2-ingredient2-step11931-tokens50B" ~ "OLMo 7B-seed2",
    model == "olmo-2-1124-7b-stage2-ingredient3-step11931-tokens50B" ~ "OLMo 7B-seed3",
    model == "olmo-2-1124-13b-stage2-ingredient1-step11931-tokens100B" ~ "OLMo 13B-seed1",
    model == "olmo-2-1124-13b-stage2-ingredient2-step11931-tokens100B" ~ "OLMo 13B-seed2",
    model == "olmo-2-1124-13b-stage2-ingredient3-step11931-tokens100B" ~ "OLMo 13B-seed3",
    model == "olmo-2-1124-7b" ~ "OLMo 7B",
    model == "olmo-2-1124-13b" ~ "OLMo 13B",
    
    model == "qwen2.5-1.5b" ~ "Qwen 1.5B",
    model == "qwen2.5-1.5b-instruct" ~ "Qwen 1.5B IT",
    model == "qwen2.5-7b" ~ "Qwen 7B",
    model == "qwen2.5-7b-instruct" ~ "Qwen 7B IT",
    model == "qwen2.5-72b" ~ "Qwen 72B",
    model == "qwen2.5-72b-instruct" ~ "Qwen 72B IT",
    
    model == "llama-3.1-8b" ~ "Llama3.1 8B",
    model == "llama-3.1-8b-instruct" ~ "Llama3.1 8B IT",
    model == "llama-3.1-70b" ~ "Llama3.1 70B",
    model == "llama-3.1-70b-instruct" ~ "Llama3.1 70B IT",
    model == "llama-3.1-405b" ~ "Llama3.1 405B",
    model == "llama-3.3-70b-instruct" ~ "Llama3.3 70B IT",
    
    model == "mistral-large-instruct-2411" ~ "Mistral 123B IT",
    
    TRUE ~ model
  ))


# remove prompt with problem in expr2
d.expr2 <- d.expr2 %>% filter(promptID!=5)


d.expr2 <- d.expr2 %>%
  mutate(dataset = case_when(
    dataset == "jabberwocky" ~ "nonsense",
    TRUE ~ dataset
  ))

d.expr2$dataset <- factor(d.expr2$dataset, 
                          levels = c("wikipedia", "news", "nonsense","randomseq"))

# model information
model_info <- d.expr1 %>%
  group_by(model) %>%
  summarise(
    model_family = first(model_family),
    model_gen      = first(model_gen),
    model_size     = first(model_size),
    model_type     = first(model_type),
    .groups = "drop"
  )

models <- unique(model_info$model)
############################analyze accuracy########################################
## experiment 1
### Plotting (Fig 2a)
d.expr1.acc = group_by(d.expr1, promptID, model, model_size, model_type) %>%
  summarise(mean.sumLP_diff = mean(sumLP_diff > 0),
            mean.prompt_diff = mean(diff > 0)) %>%
  pivot_longer(cols = c("mean.sumLP_diff", "mean.prompt_diff"), 
               names_to=c("variable")) %>%
  mutate(variable = ifelse(grepl("sumLP", variable), "log prob", "prompt"),
         promptID = ifelse(variable == "log prob", 1, promptID)) %>%
  unique()
d.expr1.acc$model = fct_reorder(d.expr1.acc$model, d.expr1.acc$model_size)
d.expr1.acc$group = paste(d.expr1.acc$variable, d.expr1.acc$promptID)

d.expr1.acc.mean <- d.expr1.acc  %>% 
  filter(variable == "prompt") %>% 
  group_by(model)  %>% 
  summarise(mean_value = mean(value),
            model_size = first(model_size),
            model_type = first(model_type))  

d.expr1.acc <- d.expr1.acc %>% 
  bind_rows(
    d.expr1.acc.mean %>% 
      mutate(variable = "prompting(mean)", group = "prompting(mean)", value = mean_value)
  )
rm(d.expr1.acc.mean)

# accuracy (Fig 2a)
ggplot(d.expr1.acc, aes(x=model, y=value, colour=variable, group=group,
                  alpha=variable, shape=model_type)) + #, ymin=l, ymax=u)) + 
  geom_point(size=3) + 
  geom_line(size=1) +
  #geom_errorbar(colour="black") + 
  theme_classic(18) +
  theme(axis.text.x = element_text(angle=45, hjust=1, size=12)) +
  scale_colour_manual(values=c("black", "orange","darkorange"),labels=c('Direct (log prob)','Meta (prompting)','Meta mean')) +
  ylim(.5, 1) +
  scale_alpha_manual(
    values = c(1, 0.3, 1),
    guide  = "none"
  ) +
  labs(
    x = "Model (by size)",    
    y = "Accuracy",            
    shape = "Model Type"       
  ) +
  theme(legend.position = 'bottom',
        legend.justification = "center",
        legend.box = "vertical",
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
        legend.spacing.x = unit(0, "lines"),
        legend.margin = margin(0, 0, 0, 0))+
  guides(
    color = guide_legend(override.aes = list(size = 3, linetype=1)),  
    shape = guide_legend(override.aes = list(size = 4))   
  )+
  geom_vline(xintercept = c(2.5, 10.5, 14.5, 20.5), color = "darkgrey", size = 0.5)

ggsave("figures/exp1_accuracy.pdf", width = 8, height = 4.5, dpi = 300)

### regression: predicting correctness from model size, evaluation method and their interaction 
d.expr1.stats = mutate(d.expr1, mean.sumLP_diff = (sumLP_diff > 0),
          mean.prompt_diff = (diff > 0)) %>%
  pivot_longer(cols = c("mean.sumLP_diff", "mean.prompt_diff"), 
               names_to=c("variable")) %>%
  mutate(variable = ifelse(grepl("sumLP", variable), "log prob", "prompt"),
         promptID = ifelse(variable == "log prob", 1, promptID),
         IsBig = model_size >= 70)

# Predicting correctness (Sec 3.2.1 & Table 5)
l.acc = glm(family="binomial",
    data = d.expr1.stats,
    value ~ variable * log(model_size)
    )
summary(l.acc)







## expr2
### overall
d.expr2.acc = group_by(d.expr2, promptID, model, model_size, model_type) %>%
  summarise(mean.sumLP_diff = mean(sumLP_diff > 0),
            mean.prompt_diff = mean(diff > 0)) %>%
  pivot_longer(cols = c("mean.sumLP_diff", "mean.prompt_diff"), 
               names_to=c("variable")) %>%
  mutate(variable = ifelse(grepl("sumLP", variable), "log prob", "prompt"),
         promptID = ifelse(variable == "log prob", 1, promptID)) %>%
  unique()
d.expr2.acc$model = fct_reorder(d.expr2.acc$model, d.expr2.acc$model_size)
d.expr2.acc$group = paste(d.expr2.acc$variable, d.expr2.acc$promptID)

d.expr2.acc.mean <- d.expr2.acc  %>% 
  filter(variable == "prompt") %>% 
  group_by(model)  %>% 
  summarise(mean_value = mean(value),
            model_size = first(model_size),
            model_type = first(model_type))  

d.expr2.acc <- d.expr2.acc %>% 
  bind_rows(
    d.expr2.acc.mean %>% 
      mutate(variable = "prompting(mean)", group = "prompting(mean)", value = mean_value)
  )
rm(d.expr2.acc.mean)

### analyze  acc / tendency of expr2 by subsets
d.expr2.acc.dataset = group_by(d.expr2, promptID, model, model_size, model_type,dataset) %>%
  summarise(mean.sumLP_diff = mean(sumLP_diff > 0),
            mean.prompt_diff = mean(diff > 0)) %>%
  pivot_longer(cols = c("mean.sumLP_diff", "mean.prompt_diff"), 
               names_to=c("variable")) %>%
  mutate(variable = ifelse(grepl("sumLP", variable), "log prob", "prompt"),
         promptID = ifelse(variable == "log prob", 1, promptID)) %>%
  unique()
d.expr2.acc.dataset$model = fct_reorder(d.expr2.acc.dataset$model, d.expr2.acc.dataset$model_size)
d.expr2.acc.dataset$group = paste(d.expr2.acc.dataset$variable, d.expr2.acc.dataset$promptID)
d.expr2.acc.dataset.mean <- d.expr2.acc.dataset  %>% 
  filter(variable == "prompt") %>% 
  group_by(model,dataset)  %>% 
  summarise(mean_value = mean(value),
            model_size = first(model_size),
            model_type = first(model_type))  

d.expr2.acc.dataset <- d.expr2.acc.dataset %>% 
  bind_rows(
    d.expr2.acc.dataset.mean %>% 
      mutate(variable = "prompting(mean)", group = "prompting(mean)", value = mean_value)
  )
rm(d.expr2.acc.dataset.mean)


ggplot(d.expr2.acc.dataset, 
       aes(x = model, 
           y = value, 
           colour = variable, 
           group = group,
           alpha = variable, 
           shape = model_type)) + 
  geom_point(size=3) + 
  geom_line(size=1) +
  facet_wrap(~ dataset, ncol = 2, scales = "fixed") +       
  theme_classic(16) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,size=10)) +
  scale_alpha_manual(values = c(1, 0.3, 1), guide = 'none') +
  scale_colour_manual(values = c("black", "orange","darkorange"), 
                      labels = c('Direct (log prob)','Meta (prompting)','Meta mean')) +
  ylim(0, 1) +
  ylab("proportion of choosing original word") + 
  xlab("Model (by size)") +
  theme(legend.position = 'bottom',
        legend.justification = "center",
        legend.box = "vertical",
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
        legend.spacing.x = unit(0, "lines"),
        legend.margin = margin(0, 0, 0, 0)
        )+
  guides(
    color = guide_legend(override.aes = list(size = 3, linetype=1)),  
    shape = guide_legend(override.aes = list(size = 4))   
  )+
  geom_vline(xintercept = c(2.5, 10.5, 14.5, 20.5), color = "darkgrey", size = 0.5)

ggsave("figures/expr2_tend_new.pdf", width = 12, height = 8, dpi = 300)



####################### do overall alignment analysis##########################

## expr 1 and expr 2 overall (Appendix B)
get_alignment <- function(d){
  d.alignment = group_by(d.expr2,promptID, model, model_size, model_type) %>%
    summarise(kappa = cohen.kappa(cbind(sumLP_ans, ans))$kappa)
  
  d.alignment$model = fct_reorder(d.alignment$model, d.alignment$model_size)
  
  p<-ggplot(d.alignment, aes(x=model, y=kappa, group=promptID,
                                   shape=model_type)) + #, ymin=l, ymax=u)) + 
    geom_point(color = "orange",size=3) + 
    geom_line(color = "orange",size=1) +
    theme_classic(16) +
    theme(
      axis.text.x = element_blank(),  
      axis.title.x = element_blank()
    ) +
    theme(axis.text.x = element_text(angle=45,hjust=1,vjust=1)) +
    ylim(-.2, .6) +
    labs(
      x = "Model (by size)",    
      y = "Cohen's Kappa",            
      shape = "Model Type"       
    )+
    theme(legend.position = c(0.1, 0.9),
          legend.title = element_blank())  +
    guides(
      color = guide_legend(override.aes = list(size = 3)),  
      shape = guide_legend(override.aes = list(size = 4)) 
    )+
    geom_vline(xintercept = c(2.5, 10.5, 14.5, 20.5), color = "darkgrey", size = 0.5)
  
  return(p)
}

# plot Fig 5
expr1.p.alignment<- get_alignment(d.expr1)
expr1.p.alignment
expr2.p.alignment<- get_alignment(d.expr2)&theme(legend.position='none')
expr2.p.alignment
ggsave("figures/expr1_cohen.pdf",expr1.p.alignment, width = 7, height = 6, dpi = 300)
ggsave("figures/expr2_cohen.pdf",expr2.p.alignment, width = 7, height = 6, dpi = 300)







################## Get filtered and unfiltered data for experiment 1################
d.expr1.unfiltered <- d.expr1

####### get rid of pairs with no disagreement or not enough
d.expr1 = group_by(d.expr1, pair_ID) %>%
  mutate(mean.sumLP_diff = mean(sumLP_diff < 0),
         mean.prompt_diff = mean(diff < 0)) %>%
  ungroup()

d.expr2 = group_by(d.expr2, pair_ID) %>%
  mutate(mean.sumLP_diff = mean(sumLP_diff < 0),
         mean.prompt_diff = mean(diff < 0)) %>%
  ungroup()

must.disagree = .05
d.expr1 = filter(d.expr1, mean.sumLP_diff >= must.disagree,
           mean.prompt_diff >= must.disagree,
           mean.sumLP_diff <= 1 - must.disagree,
           mean.prompt_diff <= 1 - must.disagree
)

## expr2 by subset
d.expr2.subsets <- split(d.expr2, d.expr2$dataset)




#################### analyze introspection with pearson r #################

## functions
###### function to get correlation matrix
get_corr_matrix_all_prompts <- function(d) {
  all_prompts <- unique(d$promptID)
  corr_result <- list()
  
  for (m1 in models) {
    for (m2 in models) {
      prompt_corr_values <- c()
      
      for (prompt in all_prompts) {
        prompt_data <- subset(d, promptID == prompt)
        
        d1 <- prompt_data %>%
          filter(model == m1) %>%
          select(pair_ID, sumLP_diff, diff) %>%
          rename(sumLP_diff1 = sumLP_diff,
                 diff1       = diff)
        
        
        d2 <- prompt_data %>%
          filter(model == m2) %>%
          select(pair_ID, sumLP_diff, diff) %>%
          rename(sumLP_diff2 = sumLP_diff,
                 diff2       = diff)
        
        
        merged_data <- inner_join(d1, d2, by = "pair_ID")
        
        if (nrow(merged_data) < 2) {
          prompt_corr <- NA
          
        } else {
          prompt_corr <- cor(merged_data$sumLP_diff1, merged_data$diff2, 
                                  method = "pearson", use = "complete.obs")
        }
        
        prompt_corr_values[as.character(prompt)] <- prompt_corr
        
        if (prompt==1){
          if (nrow(merged_data) < 2) {
            prob_corr <- NA
          } else {
            prob_corr <- cor(merged_data$sumLP_diff1, merged_data$sumLP_diff2,
                             method = "pearson", use = "complete.obs")
          }
        }
      }
      
      mean_prompt_corr <- mean(prompt_corr_values, na.rm = TRUE)
      
      info1 <- model_info %>% filter(model == m1)
      info2 <- model_info %>% filter(model == m2)
      if (m1 == m2) {
        match_type <- "self"
      } else if ((info1$model_family == info2$model_family) &&
                 (info1$model_gen      == info2$model_gen) &&
                 (info1$model_size     == info2$model_size) &&
                 (info1$model_type     == info2$model_type)) {
        match_type <- "seed variant"
      } else if ((info1$model_family == info2$model_family) &&
                 (info1$model_gen      == info2$model_gen) &&
                 (info1$model_size     == info2$model_size) &&
                 (info1$model_type     != info2$model_type)) {
        match_type <- "base/instruct"
      } else if (info1$model_family == info2$model_family) {
        match_type <- "same family"
      } else {
        match_type <- "other"
      }
      
      result_row <- tibble(
        prob_model            = m1,
        prompt_model            = m2,
        prob_size           = info1$model_size,
        prompt_size           = info2$model_size,
        match_type        = match_type,
        mean_prompt_corr  = mean_prompt_corr,
        prob_corr = prob_corr
      )
      
      prompt_cols <- as_tibble(as.list(prompt_corr_values))
      result_row <- bind_cols(result_row, prompt_cols)
      
      corr_result[[length(corr_result) + 1]] <- result_row
    }
  }
  
  result_df <- bind_rows(corr_result)
  result_df$match_type <- factor(result_df$match_type, 
                                 levels = c("self", "seed variant", "base/instruct", "same family", "other"))
  
  return(result_df)
}



###### function for scatter plot
plot_prompt_prob_corr <- function(data) {
  ggplot(data, aes(x = prob_corr, y = mean_prompt_corr, color = match_type )) +
    geom_point( ) +
    #geom_smooth(aes(group = 1), method = "lm", se = FALSE, color='black') +
  labs(title = element_blank() , x = expression(Delta ~ Direct[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)'), y = expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)')) +
    theme_classic(18) +
    theme(
      legend.position = "right",
      legend.title = element_blank(),
      legend.text = element_text(size = 18),
      legend.spacing = unit(4.0, 'cm'),
      axis.line = element_line(color = "black", size = 1),
      panel.grid.major = element_blank()
    )+
    scale_color_manual(values=sim_colors)+
    geom_smooth(aes(group = 1),  se = FALSE, color='black', lty=1, size=.5, alpha=.5)+
    guides(color=guide_legend(nrow=5,byrow=TRUE,override.aes = list(size = 4)))
}

## expr 1
### filtered
expr1.corr.all <- get_corr_matrix_all_prompts(d.expr1)



#### scatter plot (figure 3)
prompt_prob.expr1 <- plot_prompt_prob_corr(expr1.corr.all)
prompt_prob.expr1
ggsave("figures/expr1_prompt_prob.pdf", prompt_prob.expr1, width = 10, height = 5, dpi = 300)



#### bar plots (figure 3)
expr1.corr.all$ProbIsBig = expr1.corr.all$prob_size >=70
expr1.corr.all$PromptIsBig = expr1.corr.all$prompt_size >=70
expr1.corr.all$BothBig = ifelse(expr1.corr.all$ProbIsBig & expr1.corr.all$PromptIsBig, 
                                "Both ≥70B",
                                "Other")

expr1.corr.all$Big = ifelse(expr1.corr.all$PromptIsBig == 
                              expr1.corr.all$ProbIsBig, "Same Size", "Different Size")

expr1.bar <- expr1.corr.all %>%
  group_by(match_type) %>%
  summarise(m=mean(mean_prompt_corr),
            se=std.error(mean_prompt_corr),
            u=m + 1.96 *se,
            l = m - 1.96*se) %>%
  ggplot(aes(x=match_type, y=m, ymin=l, ymax=u,fill=match_type)) + 
  scale_fill_manual(values=sim_colors)+
  geom_bar(stat="identity") + 
  geom_errorbar(width=.4) +
  theme_classic(18) +
  ylab(expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)')) + 
  xlab("") +
  theme(axis.text.x = element_text(angle=45, hjust=1,size=18,colour = sim_colors,face='bold'),
        axis.title.y = element_text(size = 17),
        legend.position= 'none')
  

expr1.bar
ggsave("figures/expr1_prompt_prob_bars.pdf", width = 4, height = 5, dpi = 300)

#### heatmap
expr1.corr.all$prob_model <- with(expr1.corr.all,
                                  factor(prob_model, levels = unique(prob_model[order(prob_size)])))
expr1.corr.all$prompt_model <- with(expr1.corr.all,
                                    factor(prompt_model, levels = unique(prompt_model[order(prompt_size)])))

ggplot(expr1.corr.all, aes(x=prob_model, y=prompt_model, fill=mean_prompt_corr)) + 
  geom_tile(size=6) +
  #scale_x_log10() + scale_y_log10() +
  scale_fill_viridis(option = "magma") +
  theme_classic(16) +
  theme(axis.text.x = element_text(angle=45, hjust=1))+
  labs(x = expression(Delta ~ 'Direct Model (by size)'), y = expression(Delta ~ 'Meta Model (by size)'), fill = "mean correlation")

ggsave("figures/expr1_heatmap_pearson.pdf", dpi = 300, height = 6, width =9) 


#### STATS
##### predict correlation with model size (Sec 3.2.1)
# estimate correlation with prob model size
l.heat1.corr = lm(data=expr1.corr.all,
                  mean_prompt_corr~log(prob_size))

summary(l.heat1.corr)


# estimate correlation with prompt model size
l.heat2.corr = lm(data=expr1.corr.all,
                  mean_prompt_corr~log(prompt_size))

summary(l.heat2.corr)

# estimate correlation with interaction (Table 6)
l.heat3.corr = lm(data=expr1.corr.all,
                  mean_prompt_corr~log(prob_size)* log(prompt_size))

summary(l.heat3.corr)


#####estimate correlation with ModelSim
# Meta ~ Direct predicted with ModelSim
l.exp1.bars = lm(data=expr1.corr.all,
   mean_prompt_corr ~ match_type)

summary(l.exp1.bars)

# Direct ~ Direct predicted with ModelSim
l.exp1.bars.prob = lm(data=expr1.corr.all,
                 prob_corr ~ match_type)

summary(l.exp1.bars.prob)

# Meta ~ Direct correlated with Direct ~ Direct
cor(expr1.corr.all$mean_prompt_corr, expr1.corr.all$prob_corr, method = 'pearson')

# Meta ~ Direct predicted with Direct ~ Direct + ModelSim (Sec 3.2.2 + Table 7)
l.1.both = lm(data=expr1.corr.all,
                      mean_prompt_corr ~ prob_corr + match_type)
summary(l.1.both)

l.1.both.big = lm(data=filter(expr1.corr.all, BothBig == "Both ≥70B"),
                     mean_prompt_corr ~ prob_corr + match_type)
summary(l.1.both.big)

###expr1-unfiltered
expr1.corr.unfiltered <- get_corr_matrix_all_prompts(d.expr1.unfiltered)
#### heatmap(Fig 6)
expr1.corr.unfiltered$prob_model <- with(expr1.corr.unfiltered,
                                         factor(prob_model, levels = unique(prob_model[order(prob_size)])))
expr1.corr.unfiltered$prompt_model <- with(expr1.corr.unfiltered,
                                           factor(prompt_model, levels = unique(prompt_model[order(prompt_size)])))
ggplot(expr1.corr.unfiltered, aes(x=prob_model, y=prompt_model, fill=mean_prompt_corr)) + 
  geom_tile(size=6) +
  #scale_x_log10() + scale_y_log10() +
  theme_classic(16) +
  scale_fill_viridis(option = "magma") +
  theme(axis.text.x = element_text(angle=45, hjust=1))+
  labs(x = expression(Delta ~ 'Direct Model (by size)'), y = expression(Delta ~ 'Meta Model (by size)'), fill = "mean correlation")

ggsave("figures/expr1_heatmap_pearson_unfiltered.pdf", dpi = 300, height = 6, width =9)   
#### scatter plot(Fig 7)
prompt_prob.expr1.unfiltered <- plot_prompt_prob_corr(expr1.corr.unfiltered)
prompt_prob.expr1.unfiltered
ggsave("figures/expr1_prompt_prob_unfiltered.pdf", prompt_prob.expr1.unfiltered, width = 10, height = 5, dpi = 300)

#### bar plot(Fig 7)
expr1.corr.unfiltered$ProbIsBig = expr1.corr.unfiltered$prob_size >=70
expr1.corr.unfiltered$PromptIsBig = expr1.corr.unfiltered$prompt_size >=70
expr1.corr.unfiltered$BothBig = ifelse(expr1.corr.unfiltered$ProbIsBig & expr1.corr.unfiltered$PromptIsBig, 
                                       "Both ≥70B",
                                       "Other")

expr1.corr.unfiltered$Big = ifelse(expr1.corr.unfiltered$PromptIsBig == 
                                     expr1.corr.unfiltered$ProbIsBig, "Same Size", "Different Size")

expr1.bar.unfiltered <- expr1.corr.unfiltered %>%
  group_by(match_type) %>%
  summarise(m=mean(mean_prompt_corr),
            se=std.error(mean_prompt_corr),
            u=m + 1.96 *se,
            l = m - 1.96*se) %>%
  ggplot(aes(x=match_type, y=m, ymin=l, ymax=u,fill=match_type)) + 
  scale_fill_manual(values=sim_colors)+
  geom_bar(stat="identity") + 
  geom_errorbar(width=.4) +
  theme_classic(18) +
  ylab(expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)')) + 
  xlab("") +
  theme(axis.text.x = element_text(angle=45, hjust=1,size=18, colour = sim_colors,face='bold'),
        axis.title.y = element_text(size = 17),
        legend.position= 'none')


expr1.bar.unfiltered
ggsave("figures/expr1_prompt_prob_bars_unfiltered.pdf", width = 4, height = 5, dpi = 300)


####STATS
### estimate correlation with prob model size 
l.heat1.corr.uf = lm(data=expr1.corr.unfiltered,
                  mean_prompt_corr~log(prob_size))

summary(l.heat1.corr.uf)


### estimate correlation with prompt model size
l.heat2.corr.uf = lm(data=expr1.corr.unfiltered,
                  mean_prompt_corr~log(prompt_size))

summary(l.heat2.corr.uf)

### estimate correlation with interaction (Table 6)
l.heat3.corr.uf = lm(data=expr1.corr.unfiltered,
                  mean_prompt_corr~log(prob_size)* log(prompt_size))

summary(l.heat3.corr.uf)

l.exp1.bars.uf = lm(data=expr1.corr.unfiltered,
                 mean_prompt_corr ~ match_type)

l.exp1.bars.prob.uf = lm(data=expr1.corr.unfiltered,
                      prob_corr ~ match_type)

expr1.corr.unfiltered$match_type = factor(expr1.corr.unfiltered$match_type,
                                   levels=c("self", "seed variant",
                                            "base/instruct",
                                            "same family",
                                            "other"))

# predict with model sim (Table 7)
l.1.both.uf = lm(data=expr1.corr.unfiltered,
              mean_prompt_corr ~ prob_corr + match_type)
summary(l.1.both.uf)

l.1.both.big.uf = lm(data=filter(expr1.corr.unfiltered, BothBig == "Both ≥70B"),
                  mean_prompt_corr ~ prob_corr + match_type)
summary(l.1.both.big.uf)



## experiment 2
###get correlation matrix
expr2.corr.all <- NULL
for (expr2.subset in d.expr2.subsets) {
  corr_df <- get_corr_matrix_all_prompts(expr2.subset)
  corr_df$subset <- expr2.subset$dataset[1]
  expr2.corr.all <- rbind(expr2.corr.all, corr_df)
}


### bars (4a)
expr2.corr.all %>%
  mutate(ProbIsBig = prob_size>=70,
         PromptIsBig = prompt_size>=70,
         BothBig = ProbIsBig & PromptIsBig,
         sizegroup = ifelse(BothBig, "Both ≥70B", "Other")) %>%
  group_by(match_type, subset) %>%
  summarise(m=mean(mean_prompt_corr),
            se=std.error(mean_prompt_corr),
            u=m + 1.96 *se,
            l = m - 1.96*se) %>%
  ggplot(aes(x=match_type, y=m, ymin=l, ymax=u, fill=match_type)) + 
  geom_bar(stat="identity") + #, fill="gray") + 
  geom_errorbar(width=.4) +
  theme_classic(18) +
  ylab(expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)')) + 
  xlab("") +
  theme(axis.text.x= element_text(angle=45, hjust=1,size=18, colour = sim_colors,face='bold'),
        legend.position="none") +
  facet_grid(. ~ subset) +
  scale_fill_manual(values = sim_colors) +
  guides(fill=guide_legend(nrow=2,byrow=TRUE))

ggsave("figures/expr2_prompt_prob_bars.pdf", width = 7, height = 5, dpi = 300)

### scatter plot (4b)
ggplot(expr2.corr.all, aes(x = prob_corr, y = mean_prompt_corr, 
                           color = match_type )) +
  facet_wrap(.~ subset, ncol=2,scales = "free") +
  geom_point( size=1) +
  labs(title = element_blank() , x = expression(Delta ~ Direct[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)'), y = expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B] ~ '(Pearson r)')) +
  theme_classic(18) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.line = element_line(color = "black", size = 1),
    panel.grid.major = element_blank(),
    legend.spacing.x = unit(0, "lines"),
    legend.margin = margin(0, 0, 0, 0)
  )+
  scale_color_manual(values=sim_colors)+
  scale_shape_manual(values=c(21,24,25,22,23))+
  guides(color=guide_legend(nrow=1,byrow=TRUE,override.aes = list(size = 5)))+
  geom_smooth(aes(group = 1),  se = FALSE, color='black', lty=1, size=.5, alpha=.5)
  

ggsave("figures/expr2_subset_prompt_prob_scatter.pdf", width = 8, height = 5, dpi = 300)                 


### STATS
expr2.corr.all$ProbIsBig = expr2.corr.all$prob_size>=70
expr2.corr.all$PromptIsBig = expr2.corr.all$prompt_size>=70
expr2.corr.all$BothBig = ifelse(expr2.corr.all$ProbIsBig & expr2.corr.all$PromptIsBig, 
                                "Both ≥70B",
                                "Other")

group_by(expr2.corr.all, subset) %>%
  summarise(m=cor(mean_prompt_corr, prob_corr))
         
l.exp2.bars = lm(data=expr2.corr.all,
                 mean_prompt_corr ~ match_type * subset)
summary(l.exp2.bars)

l.exp2.bars.prob = lm(data=expr2.corr.all,
                      prob_corr ~ match_type * subset)
summary(l.exp2.bars.prob)

###heatmaps (Fig 9)
expr2.corr.all$prob_model <- with(expr2.corr.all,
                                  factor(prob_model, levels = unique(prob_model[order(prob_size)])))
expr2.corr.all$prompt_model <- with(expr2.corr.all,
                                    factor(prompt_model, levels = unique(prompt_model[order(prompt_size)])))

expr2.corr.all <- expr2.corr.all %>%
  group_by(subset) %>%
  mutate(corr.mean = mean(mean_prompt_corr, na.rm = TRUE)) %>%
  ungroup()

expr2.corr.all$corr.diff <- expr2.corr.all$mean_prompt_corr - expr2.corr.all$corr.mean

ggplot(expr2.corr.all, aes(x=prob_model, y=prompt_model, fill=corr.diff)) + 
  geom_tile(size=6) +
  #scale_x_log10() + scale_y_log10() +
  theme_classic(18) +
  scale_fill_viridis(option = "magma") +
  theme(axis.text.x = element_text(angle=45, hjust=1),legend.position = 'bottom')+
  labs(x = expression(Delta ~ 'Direct Model (by size)'), y = expression(Delta ~ 'Meta Model (by size)'), fill = "pearson r difference from dataset mean")+
  facet_wrap( .~ subset, ncol = 2)+
  guides(fill = guide_colorbar(barwidth = 15, barheight = 1))



ggsave("figures/expr2_heatmap_pearson.pdf", dpi = 300, width = 16, height =14)  


expr2.corr.all$match_type = factor(expr2.corr.all$match_type,
                                   levels=c("self", "seed variant",
                                            "base/instruct",
                                            "same family",
                                            "other"))

# predict with model similarity (Table 7)
l.2.both = lm(data=expr2.corr.all,
              mean_prompt_corr ~ prob_corr * subset + match_type * subset)
summary(l.2.both)

l.2.both.big = lm(data=filter(expr2.corr.all, BothBig == "Both ≥70B"),
                  mean_prompt_corr ~ prob_corr * subset + match_type * subset)
summary(l.2.both.big)


l.2.both.wikipedia <- lm(data=filter(expr2.corr.all,
                       subset == "wikipedia"),
           mean_prompt_corr ~ prob_corr  + match_type )

l.2.both.news <-lm(data=filter(expr2.corr.all, 
                       subset == "news"),
           mean_prompt_corr ~ prob_corr  + match_type )

l.2.both.nonsense <-lm(data=filter(expr2.corr.all,
                       subset == "nonsense"),
           mean_prompt_corr ~ prob_corr  + match_type )

l.2.both.randomseq <-lm(data=filter(expr2.corr.all, 
                       subset == "randomseq"),
           mean_prompt_corr ~ prob_corr  + match_type )
l.2.both.wikipedia.big <- lm(data=filter(expr2.corr.all, subset == "wikipedia", BothBig == "Both ≥70B"),
                         mean_prompt_corr ~ prob_corr  + match_type )


l.2.both.news.big <-lm(data=filter(expr2.corr.all, 
                               subset == "news", BothBig == "Both ≥70B"),
                   mean_prompt_corr ~ prob_corr  + match_type )

l.2.both.nonsense.big <-lm(data=filter(expr2.corr.all,
                                   subset == "nonsense", BothBig == "Both ≥70B"),
                       mean_prompt_corr ~ prob_corr  + match_type )

l.2.both.randomseq.big <-lm(data=filter(expr2.corr.all, 
                                    subset == "randomseq", BothBig == "Both ≥70B"),
                        mean_prompt_corr ~ prob_corr  + match_type )

# predict with model size (Table 6)
l.size.wiki = lm(data=filter(expr2.corr.all, subset=='wikipedia'),
                  mean_prompt_corr~log(prob_size)* log(prompt_size))

l.size.news = lm(data=filter(expr2.corr.all, subset=='news'),
           mean_prompt_corr~log(prob_size)* log(prompt_size))

l.size.nonsense = lm(data=filter(expr2.corr.all, subset=='nonsense'),
           mean_prompt_corr~log(prob_size)* log(prompt_size))

l.size.randomseq =lm(data=filter(expr2.corr.all, subset=='randomseq'),
           mean_prompt_corr~log(prob_size)* log(prompt_size))






## Appendix F: analyze olmo models (bar charts)
### Get Dataframes for olmo models (10a)
expr1.corr.olmo = expr1.corr.all %>%
  filter(grepl("OLMo", prob_model) & grepl("OLMo", prompt_model))
expr2.corr.olmo = expr2.corr.all %>%
  filter(grepl("OLMo", prob_model) & grepl("OLMo", prompt_model))
expr1.corr.olmo.unfiltered = expr1.corr.unfiltered %>%
  filter(grepl("OLMo", prob_model) & grepl("OLMo", prompt_model))


### Expr 1 (Filtered) (10a)
expr1.corr.olmo  <- expr1.corr.olmo  %>%
  mutate(
    prompt_model = recode(prompt_model,
                          "OLMo 7B-seed1" = "7B-seed1",
                          "OLMo 7B-seed2" = "7B-seed2",
                          "OLMo 7B-seed3" = "7B-seed3",
                          "OLMo 13B-seed1" = "13B-seed1",
                          "OLMo 13B-seed2" = "13B-seed2",
                          "OLMo 13B-seed3" = "13B-seed3"),
    prob_model = recode(prob_model,
                          "OLMo 7B-seed1" = "7B-seed1",
                          "OLMo 7B-seed2" = "7B-seed2",
                          "OLMo 7B-seed3" = "7B-seed3",
                          "OLMo 13B-seed1" = "13B-seed1",
                          "OLMo 13B-seed2" = "13B-seed2",
                          "OLMo 13B-seed3" = "13B-seed3"),
    match_type = recode(match_type,
                        "same family" = "different size")
  )

expr1.corr.olmo$match_type <- factor(expr1.corr.olmo$match_type, 
                                     levels = c("self", "seed variant", "different size"))

ggplot(expr1.corr.olmo, aes(x = reorder(prompt_model, prompt_size),    
                             y = mean_prompt_corr,  
                             fill = match_type,
                            group = prob_model)) +   
  geom_bar(stat = "identity",
           position = position_dodge(),
           color = "white") +  
  scale_fill_manual(values = olmo_colors) +
  theme_classic(10) +
  theme(axis.text.x = element_text(size = 8, angle=45, hjust = 1),
        legend.position = "none") +
  labs(x = expression(Delta ~ Meta ~ "Model (by size)"), y = expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B]), fill = "matchtype")

ggsave('figures/expr1_olmo_bars_pearson.pdf', height = 2, width = 8, dpi=300)

### Expr1 (Unfiltered) (10b)
expr1.corr.olmo.unfiltered  <- expr1.corr.olmo.unfiltered  %>%
  mutate(
    prompt_model = recode(prompt_model,
                          "OLMo 7B-seed1" = "7B-seed1",
                          "OLMo 7B-seed2" = "7B-seed2",
                          "OLMo 7B-seed3" = "7B-seed3",
                          "OLMo 13B-seed1" = "13B-seed1",
                          "OLMo 13B-seed2" = "13B-seed2",
                          "OLMo 13B-seed3" = "13B-seed3"),
    prob_model = recode(prob_model,
                        "OLMo 7B-seed1" = "7B-seed1",
                        "OLMo 7B-seed2" = "7B-seed2",
                        "OLMo 7B-seed3" = "7B-seed3",
                        "OLMo 13B-seed1" = "13B-seed1",
                        "OLMo 13B-seed2" = "13B-seed2",
                        "OLMo 13B-seed3" = "13B-seed3"),
    match_type = recode(match_type,
                        "same family" = "different size")
  )

expr1.corr.olmo.unfiltered$match_type <- factor(expr1.corr.olmo.unfiltered$match_type, 
                                     levels = c("self", "seed variant", "different size"))

ggplot(expr1.corr.olmo.unfiltered, aes(x = reorder(prompt_model, prompt_size),    
                            y = mean_prompt_corr,  
                            fill = match_type,
                            group = prob_model)) +   
  geom_bar(stat = "identity",
           position = position_dodge(),
           color = "white") +  
  scale_fill_manual(values = olmo_colors) +
  theme_classic(10) +
  theme(axis.text.x = element_text(size = 8, angle=45, hjust = 1),
        legend.position = "none") +
  labs(x = expression(Delta ~ Meta ~ "Model (by size)"), y = expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B]), fill = "matchtype")

ggsave('figures/expr1_olmo_bars_pearson_unfilterd.pdf', height = 2, width = 8, dpi=300)
#### Stats
expr1.corr.olmo$IsSelf <- expr1.corr.olmo$match_type == 'self'
l.expr1.olmo = lm(data = expr1.corr.olmo, mean_prompt_corr ~ IsSelf)
summary(l.expr1.olmo)

expr1.corr.olmo.unfiltered$IsSelf <- expr1.corr.olmo.unfiltered$match_type == 'self'
l.expr1.olmo.uf = lm(data = expr1.corr.olmo.unfiltered, mean_prompt_corr ~ IsSelf)
summary(l.expr1.olmo.uf)

### Expr2 (10c)
expr2.corr.olmo  <- expr2.corr.olmo  %>%
  mutate(
    prompt_model = recode(prompt_model,
                          "OLMo 7B-seed1" = "7B-seed1",
                          "OLMo 7B-seed2" = "7B-seed2",
                          "OLMo 7B-seed3" = "7B-seed3",
                          "OLMo 13B-seed1" = "13B-seed1",
                          "OLMo 13B-seed2" = "13B-seed2",
                          "OLMo 13B-seed3" = "13B-seed3"),
    prob_model = recode(prob_model,
                        "OLMo 7B-seed1" = "7B-seed1",
                        "OLMo 7B-seed2" = "7B-seed2",
                        "OLMo 7B-seed3" = "7B-seed3",
                        "OLMo 13B-seed1" = "13B-seed1",
                        "OLMo 13B-seed2" = "13B-seed2",
                        "OLMo 13B-seed3" = "13B-seed3"),
    match_type = recode(match_type,
                        "same family" = "different size")
  )

expr2.corr.olmo$match_type <- factor(expr2.corr.olmo$match_type, 
                                                levels = c("self", "seed variant", "different size"))

ggplot(expr2.corr.olmo, aes(x = reorder(prompt_model, prompt_size),    
                                       y = mean_prompt_corr,  
                                       fill = match_type,
                                       group = prob_model)) +   
  geom_bar(stat = "identity",
           position = position_dodge(),
           color = "white") +  
  scale_fill_manual(values = olmo_colors) +
  theme_classic(10) +
  theme(axis.text.x = element_text(size = 10, angle=45, hjust = 1),
        legend.position = "bottom", legend.key.size = unit(0.8,"line")) +
  labs(x = expression(Delta ~ Meta ~ "Model (by size)"), y = expression(Delta ~ Meta[A] ~ '~' ~ Delta ~ Direct[B]), fill = "matchtype")+
  facet_wrap(.~subset, ncol=1)

ggsave('figures/expr2_olmo_bars_pearson.pdf', height = 8, width = 8, dpi=300)

# stats
expr2.corr.olmo$IsSelf <- expr2.corr.olmo$match_type == 'self'
l.expr2.olmo = lm(data = expr2.corr.olmo, mean_prompt_corr ~ IsSelf)
summary(l.expr2.olmo)

l.olmo.wiki = lm(data = filter(expr2.corr.olmo, subset == 'wikipedia'), mean_prompt_corr ~ IsSelf)
l.olmo.news = lm(data = filter(expr2.corr.olmo, subset == 'news'), mean_prompt_corr ~ IsSelf)
l.olmo.nonsense = lm(data = filter(expr2.corr.olmo, subset == 'nonsense'), mean_prompt_corr ~ IsSelf)
l.olmo.randomseq = lm(data = filter(expr2.corr.olmo, subset == 'randomseq'), mean_prompt_corr ~ IsSelf)


# save stats results to latex tables
texreg(l.acc, file = "lm/methodsize.tex", booktabs = TRUE,dcolumn = TRUE,include.nobs =FALSE, include.rsquared = FALSE, include.adjrs = FALSE, custom.model.names = c('estimation'), custom.coef.names = c("Intercept","Meta", "log model size", "Meta:log model "), digits = 4)
texreg(list(l.heat3.corr,l.heat3.corr.uf, l.size.wiki, l.size.news, l.size.nonsense, l.size.randomseq), file = "lm/modelsize.tex", booktabs = TRUE,dcolumn = TRUE,include.nobs =FALSE, include.rsquared = FALSE, include.adjrs = FALSE, custom.model.names = c('exp1(filtered)', 'exp1(unfiltered)', 'wikipedia', 'news', 'nonsense', 'randomseq'), custom.coef.names = c("Intercept","log Direct model size", "log Meta model size", "Interaction"), digits = 4)
texreg(list(l.1.both,l.1.both.uf,l.2.both.wikipedia,l.2.both.news,l.2.both.nonsense,l.2.both.randomseq), file = "lm/modelsim.tex", booktabs = TRUE,dcolumn = TRUE,include.nobs =FALSE, include.rsquared = FALSE, include.adjrs = FALSE, custom.model.names = c('exp1(filtered)', 'exp1(unfiltered)', 'wikipedia', 'news', 'nonsense', 'randomseq'), custom.coef.names = c("Intercept","directdirect", "seedvariant", "baseinstruct","samefamily","other"), digits =4)
texreg(list(l.1.both.big,l.1.both.big.uf,l.2.both.wikipedia.big,l.2.both.news.big,l.2.both.nonsense.big,l.2.both.randomseq.big), file = "lm/bigmodelsim.tex", booktabs = TRUE,dcolumn = TRUE,include.nobs =FALSE, include.rsquared = FALSE, include.adjrs = FALSE, custom.model.names = c('exp1(filtered)', 'exp1(unfiltered)', 'wikipedia', 'news', 'nonsense', 'randomseq'), custom.coef.names = c("Intercept","directdirect", "baseinstruct","samefamily","other"), digits =4)
texreg(list(l.expr1.olmo,l.expr1.olmo.uf,l.olmo.wiki,l.olmo.news,l.olmo.nonsense,l.olmo.randomseq), file = "lm/olmo.tex", booktabs = TRUE,dcolumn = TRUE,include.nobs =FALSE, include.rsquared = FALSE, include.adjrs = FALSE, custom.model.names = c('exp1(filtered)', 'exp1(unfiltered)', 'wikipedia', 'news', 'nonsense', 'randomseq'), custom.coef.names = c("Intercept","self"), digits =4)

       