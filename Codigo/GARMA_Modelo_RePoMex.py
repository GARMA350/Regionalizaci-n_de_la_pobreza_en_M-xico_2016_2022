#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Funciones para el analisis del proyecto Regionalización de la pobreza multidimensional en México 2016-2022
# Autor: @Luis Armando García Rodríguez (GARMA) 

def BASES_NAN(Base,año):
  NAN_de_base = pd.DataFrame(Base.isna().sum())
  NAN_de_base = NAN_de_base[NAN_de_base[0]>0]
  print(f"------------------------------------------------------------------------------------")
  print(f"El numero de varaibles con NAN en la Base{año} es de {len(NAN_de_base)} de {len(Base.columns)} del total de Base{año} ")
  print("-------------------------------------------------------------------------------------")
  print(NAN_de_base)
  return NAN_de_base

def df_NAN(Bases):
 nan_count = Bases[0].isna().sum()
 NAN = pd.DataFrame({'Variable': nan_count.index, 'NAN': nan_count.values})
 for i in Bases[1:]:
   nan_count = i.isna().sum()
   NAN_A = pd.DataFrame({'Variable': nan_count.index, 'NAN': nan_count.values})
   NAN = pd.merge(NAN,NAN_A,on="Variable")

 NAN.set_index("Variable",inplace=True)
 NAN.columns = ['2016', '2018', '2020', '2022']
 return NAN

def variables_en_comun(l1,l2,l3,l4):
    sl1 = set(l1)
    sl2 = set(l2)
    sl3 = set(l3)
    sl4 = set(l4)

    variables_comunes = list(sl1.intersection(sl2,sl3,sl4))
    return variables_comunes

def NANS(Bases):
    N = df_NAN(Bases)
    Con_NAN = N.loc[(N["2016"] != 0) & (N["2018"] != 0) & (N["2020"] != 0) & (N["2022"] != 0)]
    Sin_NAN = N.loc[(N["2016"] == 0) & (N["2018"] == 0) & (N["2020"] == 0) & (N["2022"] == 0)]
    lista_con_nan = list(Con_NAN.index)
    lista_sin_nan = list(Sin_NAN.index)
    lista_ = lista_con_nan + lista_sin_nan

    no_esta = list()
    for i in Variables_Comunes:
        r = i in lista_
        if r == False:
            no_esta.append(i)
    return N,Con_NAN,Sin_NAN,no_esta

def limpiar(archivo,varaible,nombre_archivo):
    base_ = pd.read_csv(archivo)
    len_primario = len(base_)
    indices_nulos = base_[base_[varaible].isna()].index
    nulos = len(indices_nulos)
    base_.drop(indices_nulos,axis=0,inplace=True)
    len_secundario = len(base_)
    base_ = base_.reset_index()
    base_.drop("index",axis=1,inplace=True)
    print(f"Las observaciones anteriores a la limpiza fueron de {len_primario}, y posterior a la limpiza de {len_secundario}")
    print(f"Se eliminaron un total de {nulos} observaciones")
    base_.to_csv(nombre_archivo,index=False)

def definir_bases(archivos):
    #Bases
    Base_16 = pd.read_csv(archivos[0])
    Base_18 = pd.read_csv(archivos[1])
    Base_20 = pd.read_csv(archivos[2])
    Base_22 = pd.read_csv(archivos[3])

    #Nombres en variables

    variables_16 = list(Base_16.columns)
    variables_18 = list(Base_18.columns)
    variables_20 = list(Base_20.columns)
    variables_22 = list(Base_22.columns)

    #Obtener las varaibles en comun

    Variables_Comunes = variables_en_comun(variables_16,variables_18,variables_20,variables_22)

    #Dataframes por cada año

    Base16 = Base_16[Variables_Comunes].copy()
    Base18 = Base_18[Variables_Comunes].copy()
    Base20 = Base_20[Variables_Comunes].copy()
    Base22 = Base_18[Variables_Comunes].copy()
    Bases = [Base16,Base18,Base20,Base22]
    return Base16, Base18, Base20, Base22, Bases, Variables_Comunes


def imputar_base_general(pre_Base1,pre_Base2,pre_Base3,pre_Base4,variables_a_imputar):
  para_imputar_base1 = pre_Base1.copy()
  para_imputar_base2 = pre_Base2.copy()
  para_imputar_base3 = pre_Base3.copy()
  para_imputar_base4 = pre_Base4.copy()
  para_imputar_base_general = pd.concat([para_imputar_base1,para_imputar_base2,para_imputar_base3,para_imputar_base4])
  para_imputar_base_general.reset_index(inplace=True)
  para_imputar_base_general.drop("index",axis=1,inplace=True)
  para_imputar_X = para_imputar_base_general.drop(variables_a_imputar,axis=1)
  print(f"La dimension de la base general X para imputar es de {para_imputar_X.shape}")
  lista_filas_na = {}
  for i in variables_a_imputar:
    filas_na = list(para_imputar_base_general[para_imputar_base_general[i].isna()].index)
    print(f"NA de la variable {i} = {len(filas_na)}")
    lista_filas_na[i] = filas_na
  return para_imputar_base_general, para_imputar_X, lista_filas_na


def entrenamiento_y_prueba(para_imputar_base_general, para_imputar_X,na_dict,variable):
    para_imputar_X = para_imputar_X.drop(na_dict[variable],axis=0)
    para_imputar_Y = para_imputar_base_general[variable].copy()
    para_imputar_Y = para_imputar_Y.drop(na_dict[variable],axis=0)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(para_imputar_X, para_imputar_Y, test_size=0.10, random_state= 44)
    print(f"X_train dimension = {x_train.shape}, y_train dimension = {y_train.shape}, x_test dimension = {x_test.shape}, y_train dimension = {y_test.shape}")
    return x_train, x_test, y_train, y_test

def modelos_para_imputar(variables_a_imputar, para_imputar_base_general, para_imputar_X, na_dict):
    mpi = {}
    for i in variables_a_imputar:
        x_train, x_test, y_train, y_test = entrenamiento_y_prueba(para_imputar_base_general, para_imputar_X, na_dict, i)
        RFC = RandomForestClassifier(n_estimators=100, random_state=44)
        RFC.fit(x_train, y_train)
        mpi[i] = RFC

    joblib.dump(mpi, 'modelos_clasificadores_para_imputar.pkl')
    return mpi


#Posible espacio por el modelo de imputacion



def di_o_pol(Base):
    dicotomicas = list()
    politomicas = list()
    continuas = list()
    intensidades = ["profun","int_pobe","int_caren","int_pob","int_vulcar"]
    for i in Base.columns:
        if i not in intensidades:
            if len(Base[i].unique()) == 2:
                dicotomicas.append(i)
            elif len(Base[i].unique()) > 2 and len(Base[i].unique()) <= 13:
                politomicas.append(i)
            else:
                continuas.append(i)

    return intensidades, dicotomicas, politomicas, continuas

def di_o_pol2(Base):
    dicotomicas = list()
    politomicas = list()
    continuas = list()
    for i in Base.columns:
        if len(Base[i].unique()) == 2:
            dicotomicas.append(i)
        elif len(Base[i].unique()) > 2 and len(Base[i].unique()) <= 13:
            politomicas.append(i)
        else:
            continuas.append(i)

    return dicotomicas, politomicas, continuas

def dividir_en_estados(Base,df_Estados):
    df_Estados_C = df_Estados.copy()
    for i in enumerate(list(df_Estados_C.keys())) :
        df_Estados_C[i[1]] = Base[Base['ent']== i[0]+1]
    return df_Estados_C

def dicotomicas_var(df, dic, df_res):
    for i in df:
        for j in dic:
            ceros = (df[i][j] == 0).sum()
            unos = (df[i][j] == 1).sum()
            prop = unos / (len(df[i][j]))
            df_res[i].append(prop)

def continuas_var(df,cont,df_res):
    for i in df:
        for j in cont:
            mediana = df[i][j].median()
            df_res[i].append(mediana)

def ordenar(df,indice):
    df = pd.DataFrame(df,index=indice)
    df = df.T
    return df

def ordinalizar_intensidades(inten,Base):
    for i in inten:
        etiq = [j for j in Base[i].unique()]
        etiqe = sorted(etiq)
        categ = list(range(len(etiqe)))
        mapeo = dict(zip(etiqe, categ))
        Base[i] = Base[i].map(mapeo)

estados = ["Aguascalientes", "Baja California", "Baja California Sur", "Campeche", "Coahuila",
    "Colima", "Chiapas", "Chihuahua", "CDMX", "Durango", "Guanajuato", "Guerrero", "Hidalgo",
    "Jalisco", "EDOMEX", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
    "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas",
    "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas"]


def segmentar(Base,dic,cont):
  df_Estados = {i:{} for i in estados}
  df_dicotomicas = {i:[] for i in estados}
  df_continuas = {i:[] for i in estados}
  df_Estados_Año = dividir_en_estados(Base,df_Estados)
  dicotomicas_var(df_Estados_Año,dic,df_dicotomicas)
  df_dicotomicas = ordenar(df_dicotomicas,dic)
  continuas_var(df_Estados_Año,cont,df_continuas)
  df_continuas = ordenar(df_continuas,cont)
  df_Año = pd.merge(df_dicotomicas, df_continuas, left_index=True, right_index=True)
  return df_Año

def estandarizar(Base):
    Estandarizar = MinMaxScaler()
    Base_est = Estandarizar.fit_transform(Base)
    Base_est = pd.DataFrame(Base_est, index=Base.index, columns=Base.columns)
    return Base_est

def PrincipalComponents(Base_est,cp):
    columnas = [f"PC{i+1}" for i in range(cp)]
    CP = SparsePCA(n_components=cp)
    CP.fit(Base_est)
    ComponentesPrincipales = CP.fit_transform(Base_est)
    ComponentesPrincipales = pd.DataFrame(ComponentesPrincipales, index= Base_est.index, columns=columnas)
    Vectores_propios = pd.DataFrame(CP.components_,columns=Base_est.columns,index=columnas)
    Vectores_propios = Vectores_propios.T
    return ComponentesPrincipales, Vectores_propios

def BaseCluster():
    Base_Cluster = pd.DataFrame(estados, columns=["Estados"])
    Base_Cluster.set_index("Estados", inplace=True)
    return Base_Cluster


def Cluster01(Matriz,Base_Cluster):
    for i in range(200):
        kmeans = KMeans(n_clusters=5,n_init=100)
        kmeans.fit(Matriz)
        etiquetas = kmeans.labels_
        Base_Cluster[f"Clstr{i}"] = kmeans.labels_
    return Base_Cluster

def Cluster02(Base_Cluster):
    kmeans = KMeans(n_clusters=5,n_init=100)
    kmeans.fit(Base_Cluster)
    etiquetas = kmeans.labels_
    Base_Cluster["Clasificacion"] = kmeans.labels_
    return Base_Cluster

def representacion(ComponentesPrincipales,Base_Cluster):
    fig = px.scatter_3d(ComponentesPrincipales,x=ComponentesPrincipales['PC1'],y=ComponentesPrincipales['PC2'],z=ComponentesPrincipales['PC3'], text = ComponentesPrincipales.index, color= Base_Cluster["Clasificacion"])
    fig.update_traces(marker=dict(size=5))
    fig.show()

def mapa_df(Base_Cluster):
    Mexico_mapa = gpd.read_file("México_Estados.shp")
    edos_nombres = ["Baja California","Baja California Sur","Nayarit","Jalisco","Aguascalientes","Guanajuato","Querétaro","Hidalgo","Michoacán","EDOMEX","CDMX","Colima","Morelos","Yucatán","Campeche","Puebla","Quintana Roo","Tlaxcala","Guerrero","Oaxaca","Tabasco","Chiapas","Sonora","Chihuahua","Coahuila","Sinaloa","Durango","Zacatecas","San Luis Potosí","Nuevo León","Tamaulipas","Veracruz"]
    Mexico_mapa["Estado"] = edos_nombres
    Mexico_mapa.set_index("Estado",inplace=True)
    Mexico_mapa['ZonaPobreza'] = Base_Cluster['Clasificacion']
    return Mexico_mapa

def mapa(Mexico_mapa):
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = 'BuGn'
    Mexico_mapa.plot(column='ZonaPobreza', cmap=cmap, linewidth=0.8, ax=ax, legend=True)
    Mexico_mapa.boundary.plot(ax=ax, color='k', linewidth=0.8)
    ax.set_title('Regionalización de la Pobreza', fontsize=16)
    plt.tight_layout()
    plt.show()

def pca_cluster(Base):
  Base_est = estandarizar(Base)
  ComponentesPrincipales,VectoresPropios = PrincipalComponents(Base_est,5)
  Base_Cluster = BaseCluster()
  Base_Cluster = Cluster01(Base_est,Base_Cluster)
  Base_Cluster = Cluster02(Base_Cluster)
  representacion(ComponentesPrincipales,Base_Cluster)
  return Base_Cluster

