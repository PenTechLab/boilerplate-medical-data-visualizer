import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import the data from medical_examination.csv
df = pd.read_csv("medical_examination.csv")

# 2 Add an overweight Column by calculating BMI first
df['overweight'] = np.where(
    (df['weight'] / ((df['height'] / 100) ** 2)) > 25,  # BMI > 25
    1,  # Overweight
    0   # Not overweight
)

# 3
# Normalize Cholesterol
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, df['cholesterol']) 
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, df['cholesterol'])
# Normalize glucose
df['gluc'] = np.where(df['gluc'] == 1, 0, df['gluc'])
df['gluc'] = np.where(df['gluc'] > 1, 1, df['gluc'])


# 4 Draw the Categorical Plot
def draw_cat_plot():
    # 5 Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )


    # 6 Group and reformat the data in df_cat to split it by cardio
    df_cat = (
        df_cat.groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='count')
    )
    

    # 7 Convert the data into long format and create a chart that shows the value counts of the categorical features uisng sns.catplot() method provided by the seaborn library
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='count',
        hue='value',
        col='cardio',
        kind='bar'
    )



    # 8 Get the figure for the output and store it in the fig variable.
    #plt.subplots_adjust(top=0.8)  # Adjust the top to fit the title
    fig.fig.suptitle('Categorical Plot of Medical Examination Data', y=1.05)
    fig.fig.tight_layout()


    # 9
    fig.savefig('catplot.png')
    return fig


# 10 Draw the Heat Map
def draw_heat_map():
    # 11 Clean the data: 
    # diastolic pressure is higher than systolic 
    # height is less than the 2.5th percentile
    # height is more than the 97.5th percentile
    # weight is less than the 2.5th percentile
    # weight is more than the 97.5th percentile
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
                 
    # 12 Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14 Set up the matplotlib figure 
    fig, ax = plt.subplots(figsize=(16, 9))

    # 15 Plot the correlation matrix using sns.heatmap() method provided by the seaborn library
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Medical Examination Data')



    # 16
    fig.savefig('heatmap.png')
    return fig
