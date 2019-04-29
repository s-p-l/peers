from flask import Flask
from flask import render_template, redirect, url_for, request, Response, make_response
# import pyexcel as pe 
import pandas as pd
import numpy as np
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Label, Range1d
from bokeh.resources import CDN
from bokeh.embed import file_html, components #,json_item
from bokeh.transform import jitter
from bokeh.models.glyphs import Annulus
import json
import joblib
from joblib import load
import sklearn
from sklearn.neighbors import KNeighborsRegressor

def create_plot(y,x, names, colors, nb_var, carac):#, plot_name
    
	source = ColumnDataSource(data = dict(
    	y = y,
    	x = x,
    	names = names,
    	colors = colors,
	),
	)

	# TOOLTIPS = [
	# 	("Name", "@names"),
	# ]

	TOOLTIPS= """

    <div style ="border-style: solid;border-width: 2px;background-color:white; color:gray; position:relative;top: 200px; 
   left: 300px; width:200px;text-align:center; ">         
        <div>
            <span style="font-size: 12px; color: gray;font-family:Source Sans Pro; text-align:center;">@names</span>
        </div>
    </div>

	"""

	p = figure(tooltips=TOOLTIPS, #x_range = [,],y_range = [-0.5,nb_var]
			x_axis_label = 'Distance', y_axis_label='Observations', title=carac)
	

	p.circle('x', 'y', size=30, source=source, line_color='#747480',color='colors', fill_alpha = 0.4, fill_color = 'colors')
			# x_axis_label_text_font_style='normal', y_axis_label_text_font_style='normal',
			# x_axis_text_font = 'sans-serif')
	

	# for j in range(nb_var):
	# 	citation = Label(x=0, y=j-0.4, #x_units='screen', y_units='screen',
	# 				text=carac[j], render_mode='css',
	# 				border_line_color='gray', border_line_alpha=0.3,
	# 				background_fill_color='white', background_fill_alpha=0.3,
	# 				text_font_size="8pt",text_font_style='italic',text_color='gray',
	# 				text_font = 'Source Sans Pro',
	# 				text_alpha = 0.8)
	# 	p.add_layout(citation)
	
	p.yaxis.axis_label_text_font_style = "normal"
	p.xaxis.axis_label_text_font_style = "normal"
	p.yaxis.axis_label_text_font = "Source Sans Pro"
	p.xaxis.axis_label_text_font = "Source Sans Pro"

	p.title.text_font = "Source Sans Pro"
	p.title.text_font_style = "bold"
	p.title.align = "center"

	#return json.dumps(json_item(p, plot_name))
	return p 


app = Flask(__name__)

#Importation des données
print("DEBUT")
data_final = pd.read_csv("app/data/Data.csv")
data_final = data_final.drop("Unnamed: 0", axis=1)

#Affichage de la page home
@app.route('/')
def home():
    return render_template('home.html')

#Données entrées par l'utilisateur sur la nouvelle observation et passage à l'étape suivante
@app.route('/new_data', methods=['GET', 'POST']) #Garder en mémoire les caractéristiques des nouvelles observations.
def new_data():
	new_data = pd.DataFrame.from_dict(dict ({'Name' : [request.form['name']],
						'Country of Headquarters' : [request.form['country']],
						'Currency': [request.form['currency']],
						'Industry Group Name': [request.form['industry_group_name']],
						'Revenue growth FY2018 (%)': [request.form['revenue_growth_FY2018']],
						'EBITDA FY2017 (EUR m)': [request.form['ebitda_FY2017']],
						'EBIT (EUR m)': [request.form['ebit']],
						'Net Income (EUR m)': [request.form['net_income']],
						'Quick Ratio': [request.form['quick_ratio']],
						'CAPEX / EBITDA FY2017': [request.form['CAPEX / EBITDA FY2017']],
						'ST Debt (EUR m)': [request.form['ST Debt (EUR m)']],
						'Net Debt FY2017 (EUR m)': [request.form['Net Debt FY2017 (EUR m)']],
						'Working Capital (EUR m)': [request.form['Working Capital (EUR m)']],
						'Debt to Equity Ratio FY2017': [request.form['Debt to Equity Ratio FY2017']],
						'Basic EPS': [request.form['Basic EPS']],
						'Total Current Assets (EUR m)': [request.form['Total Current Assets (EUR m)']],
						'Cash flow from investing (EUR m)': [request.form['Cash flow from investing (EUR m)']],
						'Cash from financing (EUR m)': [request.form['Cash from financing (EUR m)']],
						'Levered Free Cash Flow (EUR m)': [request.form['Levered Free Cash Flow (EUR m)']],
						'Momentum FY2017': [request.form['Momentum FY2017']],
						'P/E Value FY2017': [request.form['P/E Value FY2017']]
						}))
	num_variables = ['Revenue growth FY2018 (%)','EBITDA FY2017 (EUR m)','EBIT (EUR m)','Net Income (EUR m)','Quick Ratio','CAPEX / EBITDA FY2017',
	'ST Debt (EUR m)','Net Debt FY2017 (EUR m)','Working Capital (EUR m)','Debt to Equity Ratio FY2017','Basic EPS','Total Current Assets (EUR m)',
	'Cash flow from investing (EUR m)','Cash from financing (EUR m)','Levered Free Cash Flow (EUR m)','Momentum FY2017','P/E Value FY2017']

	for element in num_variables:
		new_data[element] = pd.to_numeric(new_data[element])

	# print(new_data.dtypes)
	app.new_data = new_data
	#new_data.to_csv("app/data/new_data.csv")
	return render_template('choice.html')


# #Affichage de la page choice
# @app.route('/choice/')#sera affiché sur la page choice
# def choice():
#     return render_template('choice.html')

#Affichage de la page automated process et résultats
@app.route('/automated_process/')#Sera affiché sur la page automated_process
def automated_process():
    
	knn = joblib.load("app/data/model.joblib")

	carac = ['Industry Group Name','Revenue growth FY2018 (%)','EBITDA FY2017 (EUR m)','Net Income (EUR m)','Quick Ratio'] #Variables optimales selon l'algorithme

	data_final_a = data_final

	data_final_a['distance_euclidienne'] = np.zeros(len(data_final_a))

	new_var = []
	for i in range(len(carac)):
		new_var.append('var_'+str(i))
		if app.new_data[carac[i]].dtypes == "O":
			data_final_a['var_'+str(i)] = [0 if data_final_a.loc[j,carac[i]] == app.new_data.loc[0, carac[i]] else 1 for j in range(len(data_final_a))]
		else:
			if max(data_final_a[carac[i]]) >= app.new_data.loc[0, carac[i]]:
				app.new_data[carac[i]] = app.new_data[carac[i]]/max(data_final_a[carac[i]])
				data_final_a['var_'+str(i)] = (data_final_a[carac[i]]/max(data_final_a[carac[i]])- app.new_data.loc[0, carac[i]])**2
			else:
				app.new_data[carac[i]] = app.new_data[carac[i]]/app.new_data.loc[0, carac[i]]
				data_final_a['var_'+str(i)] = (data_final_a[carac[i]]/app.new_data.loc[0, carac[i]]- 1.0)**2
		data_final_a['distance_euclidienne'] = data_final_a['distance_euclidienne'] + data_final_a['var_'+str(i)]

	data_final_a['distance_euclidienne'] = np.sqrt(data_final_a['distance_euclidienne']/len(carac))

	res = pd.DataFrame(data_final_a['distance_euclidienne'].nsmallest(10))
	print(res)
	res_names = list(data_final_a['Name'].iloc[list(res.index)])	
	print(res_names)
	print(list(data_final_a['Industry Group Name'].iloc[list(res.index)]))

	dummies = pd.read_csv("app/data/dummies_name.csv")
	dummies = dummies.drop('Unnamed: 0',axis=1)
	new_data_transformed = app.new_data[carac]
	var_x_discrete = ['Industry Group Name']
	# for i in range(new_data_transformed.shape[1]):
	# 	if new_data_transformed.dtypes[i] == 'O':
	# 		var_x_discrete.append(new_data_transformed.columns[i])

	if len(var_x_discrete)>0:
		x_int = pd.get_dummies(new_data_transformed[var_x_discrete], prefix=None, prefix_sep = '_', dummy_na=False, columns=None)
		# print(x_int.columns,len(x_int.columns))
		for i in range(len(x_int.columns)):
			for j in range(len(dummies)):
				# print(i,j,x_int.columns[i],dummies.loc[j,'text'])
				if x_int.columns[i] == dummies.loc[j,'text']:
					x_int[dummies.loc[j,'text']] = 1
					x_int = x_int.drop(x_int.columns[i], axis=1)
					break
		for j in range(len(dummies)):
			if dummies.loc[j,'text'] not in x_int.columns:
				x_int[dummies.loc[j,'text']] = 0
    
		for element in x_int.columns:
			if element not in list(dummies['text']):
				x_int = x_int.drop(element, axis=1)
            
		X = pd.concat([new_data_transformed.drop(var_x_discrete, axis=1), x_int], axis=1, sort = False)


	# entreprise_value = knn.predict(X)[0]
	peers_distances = knn.kneighbors(X)
	print(peers_distances, "distances")

	res_names = data_final_a.loc[peers_distances[1][0],'Name']
	data_final_a['Peers'] = [1 if j in peers_distances[1][0] else 0 for j in range(len(data_final_a))]

	liste = data_final_a['var_0']#/max(data_final_a['var_0'])
	for i in range(1, len(carac)):
		liste = liste.append(data_final_a['var_'+str(i)])#/max(data_final_a['var_'+str(i)])

	data_ordinary_a = data_final_a[data_final_a['Peers']==0] #Comparables et non comparables
	data_peers_a = data_final_a[data_final_a['Peers']==1]

	perf1 = round(data_peers_a['erreur'].iloc[0]*100,2)
	perf2 = round(data_peers_a['erreur'].iloc[1]*100,2)
	perf3 = round(data_peers_a['erreur'].iloc[2]*100,2)
	perf4 = round(data_peers_a['erreur'].iloc[3]*100,2)
	perf5 = round(data_peers_a['erreur'].iloc[4]*100,2)
	perf = round((perf1 + perf2 + perf3 + perf4 + perf5)/5,2)

	data_reduced_a = data_peers_a.append(data_ordinary_a.iloc[np.random.randint(len(data_ordinary_a), size=10)]) #Moindre quantité de données
	liste_reduced_a =data_reduced_a['var_0']#/max(data_reduced_a['var_0'])
	for i in range(1, len(carac)):
		liste_reduced_a = liste_reduced_a.append(data_reduced_a['var_'+str(i)])#/max(data_reduced_a['var_'+str(i)])
	
	peers_colors = ["#ffe260" if data_reduced_a['Peers'].iloc[j] ==1 else '#747480' for j in range(len(data_reduced_a))]

	# y = np.array(np.repeat(range(len(carac)),np.repeat(len(data_reduced_a),len(carac))))
	# x = np.array(liste_reduced_a)
	# names = np.array(list(data_reduced_a['Name'])*len(carac))
	# colors = np.array(peers_colors*len(carac)) 
	# print(app.new_data.dtypes)
	# print(y,x, names, colors)

	# plot = create_plot(y=y,x=x,names=names, colors=colors, nb_var=len(carac), carac = carac)#,plot_name = 'manual_plot'

	# script, div = components(plot)

	names = data_reduced_a['Name']
	colors = peers_colors

	dict_results = dict()
	for i in range(len(carac)):
		y = list(range(1,len(data_reduced_a[new_var[i]])+1))
		x = data_reduced_a[new_var[i]]
		plot = create_plot(y=y,x=x,names=names, colors=colors, nb_var=len(carac), carac = carac[i])#,plot_name = 'manual_plot'
		# script = script + components(plot)[0]
		# div = div + components(plot)[1]
		# script = components(plot)[1]
		dict_results[carac[i]] = plot#script, div

	script, div = components(tuple(dict_results.values()))
	div = [div[j].replace('\n','') for j in range(len(div))]
	flat_div = ''
	for j in range(len(div)):
		flat_div = flat_div + div[j]


	peer1 = res_names.iloc[0]
	peer2 = res_names.iloc[1]
	peer3 = res_names.iloc[2]
	peer4 = res_names.iloc[3]
	peer5 = res_names.iloc[4]
	# peer6 = res_names.iloc[5]
	# peer7 = res_names.iloc[6]
	# peer8 = res_names.iloc[7]
	# peer9 = res_names.iloc[8]
	# peer10 = res_names.iloc[9]
 	
	return render_template('automated_process.html', 
	peer1 = peer1,
	peer2 = peer2, 
	peer3 = peer3,
	peer4 = peer4,
	peer5 = peer5,
	# peer6 = peer6,
	# peer7 = peer7,
	# peer8 = peer8,
	# peer9 = peer9,
	# peer10 = peer10,
	script_automated = script, 
	div_automated = flat_div,
	perf_automated = perf,
	perf1 = perf1, perf2=perf2, perf3=perf3, perf4=perf4, perf5=perf5)#


#Affichage de la page selected_variables
@app.route('/selected_variables/')#Sera affiché sur la page selected_variables
def selected_variables():
    return render_template('selected_variables.html')



#Données entrées par l'utilisateur sur le choix des variables et passage à l'étape suivante
@app.route('/selection', methods=['GET', 'POST']) 
def selection():
	selection = request.form.getlist('variables')
	#print(request.form.getlist('variables'))
	#selection.to_csv('app/data/selection.txt')
	#app.selection = selection
	#print(selection)

	#carac = list(selection['0'])

	nb_var = len(selection) #Nombre de variables sélectionnées par l'utilisateur

	data_final_b = data_final
	print(data_final_b.head())
	data_final_b['distance_euclidienne'] = np.zeros(len(data_final_b))

	new_var = []
	for i in range(nb_var):
		new_var.append('var_'+str(i))
		if app.new_data[selection[i]].dtypes == "O":
			data_final_b['var_'+str(i)] = [0 if data_final_b.loc[j,selection[i]] == app.new_data.loc[0, selection[i]] else 1 for j in range(len(data_final_b))]
		else:
			if max(data_final_b[selection[i]])>= app.new_data.loc[0, selection[i]]:
				data_final_b['var_'+str(i)] = (data_final_b[selection[i]]/max(data_final_b[selection[i]])- app.new_data.loc[0, selection[i]]/max(data_final_b[selection[i]]))**2
			else:
				data_final_b['var_'+str(i)] = (data_final_b[selection[i]]/app.new_data.loc[0, selection[i]]- 1.0)**2

		data_final_b['distance_euclidienne'] = data_final_b['distance_euclidienne'] + data_final_b['var_'+str(i)]

	data_final_b['distance_euclidienne'] = np.sqrt(data_final_b['distance_euclidienne']/nb_var)


	res = pd.DataFrame(data_final_b['distance_euclidienne'].nsmallest(10))
	res_names = list(data_final_b['Name'].iloc[list(res.index)])
	data_final_b['Peers'] = [1 if j in list(res.index) else 0 for j in range(len(data_final_b))]

	# liste = data_final_b['var_0']#/max(data_final_b['var_0'])
	# for i in range(1, nb_var):
	# 	liste = liste.append(data_final_b['var_'+str(i)])#/max(data_final_b['var_'+str(i)])

	data_ordinary_b = data_final_b[data_final_b['Peers']==0] #Comparables et non comparables
	data_peers_b = data_final_b[data_final_b['Peers']==1]

	# perf = round((1 - np.mean(data_peers_b['distance_euclidienne']))*100,2)
	perf1 = round(data_peers_b['distance_euclidienne'].iloc[0]*100,2)
	perf2 = round(data_peers_b['distance_euclidienne'].iloc[1]*100,2)
	perf3 = round(data_peers_b['distance_euclidienne'].iloc[2]*100,2)
	perf4 = round(data_peers_b['distance_euclidienne'].iloc[3]*100,2)
	perf5 = round(data_peers_b['distance_euclidienne'].iloc[4]*100,2)
	perf6 = round(data_peers_b['distance_euclidienne'].iloc[5]*100,2)
	perf7 = round(data_peers_b['distance_euclidienne'].iloc[6]*100,2)
	perf8 = round(data_peers_b['distance_euclidienne'].iloc[7]*100,2)
	perf9 = round(data_peers_b['distance_euclidienne'].iloc[8]*100,2)
	perf10 = round(data_peers_b['distance_euclidienne'].iloc[9]*100,2)

	data_reduced_b = data_peers_b.append(data_ordinary_b.iloc[np.random.randint(len(data_ordinary_b), size=5)]) #Moindre quantité de données
	# liste_reduced_b =data_reduced_b['var_0']#/max(data_reduced_b['var_0']))
	# for i in range(1, nb_var):
	# 	liste_reduced_b = liste_reduced_b.append(data_reduced_b['var_'+str(i)])#/max(data_reduced_b['var_'+str(i)])

	peers_colors = ["#ffe260" if data_reduced_b['Peers'].iloc[j] ==1 else '#747480' for j in range(len(data_reduced_b))]

	# y = np.array(np.repeat(range(nb_var),np.repeat(len(data_reduced_b),nb_var)))
	# x = np.array(liste_reduced_b)
	# names = np.array(list(data_reduced_b['Name'])*nb_var)
	# colors = np.array(peers_colors*nb_var) 

	names = data_reduced_b['Name']
	colors = peers_colors

	dict_results = dict()
	for i in range(nb_var):
		y = list(range(1,len(data_reduced_b[new_var[i]])+1))
		x = data_reduced_b[new_var[i]]
		plot = create_plot(y=y,x=x,names=names, colors=colors, nb_var=nb_var, carac = selection[i])#,plot_name = 'manual_plot'
		# script = script + components(plot)[0]
		# div = div + components(plot)[1]
		# script = components(plot)[1]
		dict_results[selection[i]] = plot#script, div

	script, div = components(tuple(dict_results.values()))
	div = [div[j].replace('\n','') for j in range(len(div))]
	flat_div = ''
	for j in range(len(div)):
		flat_div = flat_div + div[j]

	peer1 = res_names[0]
	peer2 = res_names[1]
	peer3 = res_names[2]
	peer4 = res_names[3]
	peer5 = res_names[4]
	peer6 = res_names[5]
	peer7 = res_names[6]
	peer8 = res_names[7]
	peer9 = res_names[8]
	peer10 = res_names[9]
  	
	return render_template('manual_process.html', 
	peer1 = peer1, peer2 = peer2, peer3 = peer3, peer4 = peer4, peer5 = peer5, peer6 = peer6, peer7 = peer7, peer8 = peer8, peer9 = peer9, peer10 = peer10,
	script_manual = script, 
	div_manual = flat_div,
	# perf_manual = perf,
	perf1 = perf1,perf2 = perf2,perf3= perf3,perf4 = perf4,perf5 = perf5, perf6=perf6, perf7=perf7,perf8=perf8,perf9=perf9, perf10=perf10)#


# #Chargement des nouvelles données et transformation pour calculer les distances

# new_data = pd.read_csv("app/data/new_data.csv", sep=',')
# new_data = new_data.drop("Unnamed: 0", axis=1)
# print(new_data)


# selected = pd.read_csv("app/data/selection.txt")
# carac = list(app.selection['0'])

# nb_var = len(carac) #Nombre de variables sélectionnées par l'utilisateur

# data_final_b = data_final

# data_final_b['distance_euclidienne'] = np.zeros(len(data_final_b))

# for i in range(nb_var):
# 	if app.new_data[app.selection.loc[i,"0"]].dtypes == "O":
# 		data_final_b['var_'+str(i)] = [1 if data_final_b.loc[j,carac[i]] == app.new_data.loc[0, carac[i]] else 0 for j in range(len(data_final_b))]
# 	else:
# 		data_final_b['var_'+str(i)] = (data_final_b[carac[i]]- app.new_data.loc[0, carac[i]])**2
# 	data_final_b['distance_euclidienne'] = data_final_b['distance_euclidienne'] + data_final_b['var_'+str(i)]

# data_final_b['distance_euclidienne'] = np.sqrt(data_final_b['distance_euclidienne'])

# res = pd.DataFrame(data_final_b['distance_euclidienne'].nsmallest(10))
# res_names = data_final_b['Name'].iloc[list(res.index)]
# data_final_b['Peers'] = [1 if j in list(res.index) else 0 for j in range(len(data_final_b))]

# liste = data_final_b['var_0']/max(data_final_b['var_0'])
# for i in range(1, nb_var):
#     liste = liste.append(data_final_b['var_'+str(i)]/max(data_final_b['var_'+str(i)]))

# data_ordinary_b = data_final_b[data_final_b['Peers']==0] #Comparables et non comparables
# data_peers_b = data_final_b[data_final_b['Peers']==1]

# data_reduced_b = data_peers_b.append(data_ordinary_b.iloc[np.random.randint(len(data_ordinary_b), size=5)]) #Moindre quantité de données
# liste_reduced_b =(data_reduced_b['var_0']/max(data_reduced_b['var_0']))
# for i in range(1, nb_var):
#     liste_reduced_b = liste_reduced_b.append((data_reduced_b['var_'+str(i)]/max(data_reduced_b['var_'+str(i)])))




if __name__ == '__main__':
    app.run(debug=True)