################################################################################
# run_gamm_noeuler_hemi.R
#
# Covaried GAMM with Euler Z, full age range
# Covariates: eTIV, Sex, Euler Z
# Tests hemisphere differences within each group
################################################################################

library(mgcv)
library(ggplot2)
library(dplyr)
library(patchwork)

SCRIPT_DIR <- "/Users/aaronalthauser/MRI_local/myMixedModelsTrajectories"
OUT_DIR <- file.path(SCRIPT_DIR, "results_gamm_trajectory")

# --------------------------------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------------------------------
df <- read.csv(file.path(SCRIPT_DIR, "all_fs_volumes_cognitive.csv"))
names(df)[names(df) == "Subject_ID"] <- "subj_id"
names(df)[names(df) == "Age"] <- "age"

for (col in c("age", "Diagnosis_bin", "clau_lh_Volume_mm3", "clau_rh_Volume_mm3",
              "measure_eTIV", "Gender_bin", "euler_z")) {
  df[[col]] <- as.numeric(df[[col]])
}
df$subj_id <- as.factor(df$subj_id)
df <- df[complete.cases(df[, c("age", "Diagnosis_bin", "clau_lh_Volume_mm3",
                                "clau_rh_Volume_mm3", "measure_eTIV",
                                "Gender_bin", "euler_z")]), ]
df <- df[df$age <= 35, ]

for (col in c("clau_lh_Volume_mm3", "clau_rh_Volume_mm3")) {
  mu <- mean(df[[col]]); sigma <- sd(df[[col]])
  df <- df[df[[col]] >= (mu - 3 * sigma) & df[[col]] <= (mu + 3 * sigma), ]
}

df$clau_total <- df$clau_lh_Volume_mm3 + df$clau_rh_Volume_mm3
df$Group <- factor(ifelse(df$Diagnosis_bin == 0, "HC", "22q"), levels = c("HC", "22q"))
df$eTIV_z <- scale(df$measure_eTIV)[, 1]

for (grp in c("HC", "22q")) {
  sub <- df[df$Group == grp, ]
  cat(grp, ": ", nrow(sub), " obs, ", length(unique(sub$subj_id)),
      " subjects, age ", round(min(sub$age), 1), "-", round(max(sub$age), 1), "\n", sep = "")
}
cat("\n")

# --------------------------------------------------------------------------
# 2. Colors and theme
# --------------------------------------------------------------------------
col_hc  <- "#2563EB"
col_22q <- "#DC2626"
fill_hc  <- "#BFDBFE"
fill_22q <- "#FECACA"

pub_theme <- theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, color = "grey40"),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
    plot.margin = margin(5, 8, 5, 5)
  )

# --------------------------------------------------------------------------
# 3. Fit covaried models (no Euler Z) per group, per hemisphere
# --------------------------------------------------------------------------
fits <- list()

for (hemi in c("Left", "Right")) {
  ycol <- ifelse(hemi == "Left", "clau_lh_Volume_mm3", "clau_rh_Volume_mm3")

  cat(strrep("-", 50), "\n")
  cat(hemi, "Claustrum\n")
  cat(strrep("-", 50), "\n")

  for (grp in c("HC", "22q")) {
    sub <- df[df$Group == grp, ]
    age_seq <- seq(min(sub$age), max(sub$age), length.out = 300)
    dummy_subj <- sub$subj_id[1]

    fit <- bam(as.formula(paste(ycol,
               "~ s(age, k=10, bs='tp') + Gender_bin + eTIV_z + euler_z + s(age, subj_id, bs='fs', m=1, k=5)")),
               data = sub, method = "fREML", discrete = TRUE)
    nd <- data.frame(age = age_seq, Gender_bin = mean(sub$Gender_bin),
                     eTIV_z = 0, euler_z = 0, subj_id = dummy_subj)

    pred <- predict(fit, newdata = nd, se.fit = TRUE, exclude = "s(age,subj_id)")
    summ <- summary(fit)

    edf <- round(summ$s.table[1, "edf"], 2)
    pval <- summ$s.table[1, "p-value"]
    rsq <- round(summ$r.sq, 3)
    fit_vals <- as.numeric(pred$fit)
    slope <- round((fit_vals[length(fit_vals)] - fit_vals[1]) /
                   (max(age_seq) - min(age_seq)), 1)
    r_val <- round(cor(age_seq, fit_vals), 3)
    peak_age <- age_seq[which.max(fit_vals)]

    cat(sprintf("  %s: slope=%.1f mm³/yr, r=%.3f, R²=%.3f, EDF=%.2f, p=%.4f\n",
                grp, slope, r_val, rsq, edf, pval))

    key <- paste0(hemi, "_", grp)
    fits[[key]] <- list(
      pred = data.frame(age = age_seq, fit = fit_vals,
                        lo = fit_vals - 1.96 * as.numeric(pred$se.fit),
                        hi = fit_vals + 1.96 * as.numeric(pred$se.fit)),
      edf = edf, rsq = rsq, slope = slope, r_val = r_val, pval = pval,
      raw_data = sub, ycol = ycol
    )
  }
  cat("\n")
}

# --------------------------------------------------------------------------
# 4. Hemisphere difference test within each group
# --------------------------------------------------------------------------
cat(strrep("=", 70), "\n")
cat("HEMISPHERE DIFFERENCE TESTS\n")
cat(strrep("=", 70), "\n\n")

# Reshape to long format: one row per hemisphere per observation
df_long <- rbind(
  data.frame(subj_id = df$subj_id, age = df$age, Group = df$Group,
             Gender_bin = df$Gender_bin, eTIV_z = df$eTIV_z, euler_z = df$euler_z,
             volume = df$clau_lh_Volume_mm3, Hemisphere = "Left"),
  data.frame(subj_id = df$subj_id, age = df$age, Group = df$Group,
             Gender_bin = df$Gender_bin, eTIV_z = df$eTIV_z, euler_z = df$euler_z,
             volume = df$clau_rh_Volume_mm3, Hemisphere = "Right")
)
df_long$Hemisphere <- factor(df_long$Hemisphere, levels = c("Left", "Right"))
df_long$Hemi_ord <- as.ordered(df_long$Hemisphere)
contrasts(df_long$Hemi_ord) <- "contr.treatment"

for (grp in c("HC", "22q")) {
  sub_long <- df_long[df_long$Group == grp, ]

  cat(grp, "— Hemisphere effect on trajectory:\n")

  # Model with hemisphere-specific smooths + random slopes
  fit_hemi <- bam(volume ~ Hemi_ord + s(age, k=10, bs="tp") +
                   s(age, by=Hemi_ord, k=10, bs="tp") +
                   Gender_bin + eTIV_z + euler_z +
                   s(age, subj_id, bs="fs", m=1, k=5),
                   data = sub_long, method = "fREML", discrete = TRUE)

  sh <- summary(fit_hemi)
  cat("  Parametric (Hemi intercept):\n")
  hemi_row <- grep("Hemi_ord", rownames(sh$p.table))
  if (length(hemi_row) > 0) {
    cat(sprintf("    Estimate = %.2f mm³, t = %.2f, p = %.4f\n",
                sh$p.table[hemi_row, "Estimate"],
                sh$p.table[hemi_row, "t value"],
                sh$p.table[hemi_row, "Pr(>|t|)"]))
  }

  cat("  Smooth (Hemi × age interaction):\n")
  hemi_smooth <- grep("Hemi_ord", rownames(sh$s.table))
  if (length(hemi_smooth) > 0) {
    cat(sprintf("    EDF = %.2f, F = %.2f, p = %.4f\n",
                sh$s.table[hemi_smooth, "edf"],
                sh$s.table[hemi_smooth, "F"],
                sh$s.table[hemi_smooth, "p-value"]))
  }
  cat("\n")
}

# --------------------------------------------------------------------------
# 5. Group difference tests (without Euler Z)
# --------------------------------------------------------------------------
cat(strrep("=", 70), "\n")
cat("GROUP DIFFERENCE TESTS (with Euler Z)\n")
cat(strrep("=", 70), "\n\n")

for (hemi in c("Left", "Right")) {
  ycol <- ifelse(hemi == "Left", "clau_lh_Volume_mm3", "clau_rh_Volume_mm3")

  df$Group_ord <- as.ordered(df$Group)
  contrasts(df$Group_ord) <- "contr.treatment"

  fit_grp <- bam(as.formula(paste(ycol,
                 "~ Group_ord + s(age, k=10, bs='tp') + s(age, by=Group_ord, k=10, bs='tp') + Gender_bin + eTIV_z + euler_z + s(age, subj_id, bs='fs', m=1, k=5)")),
                 data = df, method = "fREML", discrete = TRUE)
  sg <- summary(fit_grp)

  cat(hemi, "Claustrum — Group difference:\n")
  grp_p <- grep("Group_ord", rownames(sg$p.table))
  if (length(grp_p) > 0) {
    cat(sprintf("  Intercept diff: %.2f mm³, t=%.2f, p=%.4f\n",
                sg$p.table[grp_p, "Estimate"],
                sg$p.table[grp_p, "t value"],
                sg$p.table[grp_p, "Pr(>|t|)"]))
  }
  grp_s <- grep("Group_ord", rownames(sg$s.table))
  if (length(grp_s) > 0) {
    cat(sprintf("  Trajectory diff: EDF=%.2f, F=%.2f, p=%.4f\n",
                sg$s.table[grp_s, "edf"],
                sg$s.table[grp_s, "F"],
                sg$s.table[grp_s, "p-value"]))
  }
  cat("\n")
}

# --------------------------------------------------------------------------
# 6. Plot: Covaried LR with stats (no Euler Z)
# --------------------------------------------------------------------------
cat("Building plots...\n")

make_panel_stats <- function(hemi) {
  hc  <- fits[[paste0(hemi, "_HC")]]
  q22 <- fits[[paste0(hemi, "_22q")]]
  ycol <- hc$ycol

  all_y <- c(hc$pred$fit, q22$pred$fit,
             hc$raw_data[[ycol]], q22$raw_data[[ycol]])
  y_max <- max(all_y, na.rm = TRUE)
  y_rng <- y_max - min(all_y, na.rm = TRUE)

  hc_label <- sprintf("HC: slope = %.1f mm³/yr\nr = %.3f, R² = %.3f, EDF = %.2f",
                       hc$slope, hc$r_val, hc$rsq, hc$edf)
  q22_label <- sprintf("22q: slope = %.1f mm³/yr\nr = %.3f, R² = %.3f, EDF = %.2f",
                        q22$slope, q22$r_val, q22$rsq, q22$edf)

  x_pos <- min(hc$pred$age) + 0.5

  p <- ggplot() +
    geom_point(data = hc$raw_data, aes(x = age, y = .data[[ycol]]),
               color = col_hc, alpha = 0.12, size = 0.5) +
    geom_point(data = q22$raw_data, aes(x = age, y = .data[[ycol]]),
               color = col_22q, alpha = 0.12, size = 0.5) +
    geom_ribbon(data = hc$pred, aes(x = age, ymin = lo, ymax = hi),
                fill = fill_hc, alpha = 0.4) +
    geom_ribbon(data = q22$pred, aes(x = age, ymin = lo, ymax = hi),
                fill = fill_22q, alpha = 0.4) +
    geom_line(data = hc$pred, aes(x = age, y = fit), color = col_hc, linewidth = 1.1) +
    geom_line(data = q22$pred, aes(x = age, y = fit), color = col_22q, linewidth = 1.1) +
    annotate("text", x = x_pos, y = y_max + y_rng * 0.02,
             label = hc_label, color = col_hc, size = 3, hjust = 0, vjust = 1,
             fontface = "bold", lineheight = 0.9) +
    annotate("text", x = x_pos, y = y_max - y_rng * 0.12,
             label = q22_label, color = col_22q, size = 3, hjust = 0, vjust = 1,
             fontface = "bold", lineheight = 0.9) +
    labs(x = "Age (years)", y = "Volume (mm³)",
         title = paste(hemi, "Claustrum")) +
    pub_theme +
    theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))

  return(p)
}

p_left  <- make_panel_stats("Left")
p_right <- make_panel_stats("Right")

# Legend
leg_df <- data.frame(x = 1:2, y = 1:2,
                     Group = factor(c("HC", "22q"), levels = c("HC", "22q")))
p_leg <- ggplot(leg_df, aes(x = x, y = y, color = Group)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = c("HC" = col_hc, "22q" = col_22q),
                     labels = c("HC" = "HC (Healthy Controls)",
                                "22q" = "22q (22q11.2 Deletion)")) +
  guides(color = guide_legend(title = NULL, override.aes = list(linewidth = 2))) +
  theme_void(base_size = 11) +
  theme(legend.position = "bottom", legend.text = element_text(size = 11))

legend_grob <- cowplot::get_legend(p_leg)

fig_lr <- (p_left | p_right) +
  plot_annotation(
    title = "Claustrum Developmental Trajectories",
    subtitle = "GAMM with covariates: eTIV, Sex, Euler Z | Random slopes per subject (factor smooth) | Shaded = 95% CI",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "grey30")
    )
  )

fig_lr_final <- cowplot::plot_grid(fig_lr, legend_grob, ncol = 1, rel_heights = c(1, 0.05))

ggsave(file.path(OUT_DIR, "gamm_covaried_LR_noeuler.png"), fig_lr_final,
       width = 12, height = 6, dpi = 300, bg = "white")
ggsave(file.path(OUT_DIR, "gamm_covaried_LR_noeuler.pdf"), fig_lr_final,
       width = 12, height = 6, bg = "white")
cat("Saved: gamm_covaried_LR_noeuler.png/pdf\n")

# --------------------------------------------------------------------------
# 7. Summary
# --------------------------------------------------------------------------
sink(file.path(OUT_DIR, "gamm_summary_noeuler.txt"))
cat(strrep("=", 70), "\n")
cat("GAMM TRAJECTORIES — COVARIATES: eTIV + SEX + EULER Z\n")
cat("Covariates: eTIV (z-scored), Sex, Euler Z\n")
cat(strrep("=", 70), "\n\n")
cat("Dataset:", nrow(df), "obs,", length(unique(df$subj_id)), "subjects\n\n")
for (hemi in c("Left", "Right")) {
  cat(strrep("-", 40), "\n")
  cat(hemi, "Claustrum\n")
  cat(strrep("-", 40), "\n")
  for (grp in c("HC", "22q")) {
    r <- fits[[paste0(hemi, "_", grp)]]
    cat(sprintf("  %s: slope=%.1f mm³/yr, r=%.3f, R²=%.3f, EDF=%.2f, p=%.4f\n",
                grp, r$slope, r$r_val, r$rsq, r$edf, r$pval))
  }
  cat("\n")
}
sink()
cat("Saved: gamm_summary_noeuler.txt\n")

cat("\nDONE\n")
