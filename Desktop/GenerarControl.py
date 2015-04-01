# /usr/bin/python
# -*- coding: utf-8 -*-
# from __future__ import division
__author__, __mail__ = 'Richard Andrey Salazar Serna', 'rass12398@hotmail.com,riansalazar@utp.edu.co'
__author2__, __mail2__ = 'Rafael Pinzon Rivera', 'rafapinzon@hotmail.com'

import MySQLdb
import datetime
import numpy as np
from coopr.pyomo import *


trayecto_est_prog_dia                   = []     # Trayecto Estacion Programación Día
velocidad_tiempo_distancia_linea        = []     # Trayecto Linea lectura
velocidad_tiempo_distancia_linea_dic    = {}     # Diccionario con idlinea, velocidad_tiempo_distancia_linea_dic[linea]=[velocidad,tiempo,distancia,estinicio,estfinal]
demanda_proporcion                      = {}     # Estación matrices: demanda, proporcion
demanda_dic                             = dict() # Diccionario con doble llave (estacion inicio, estacion final) para demanda y proporcion, Tiempo de abordaje y salida de buses
servicio_holding                        = dict() # Diccionario para los holding de cada servicio, si no aparece en este diccionario el holding es 0.

### Necesario para crear hora llegada estimada.
def Conversion_lista():
    '''
    Funcion para hacer de la tupla tabla_trayecto_est_prog_dia una lista que se pueda modificar.
    '''
    global tabla_trayecto_est_prog_dia
    tabla_trayecto_est_prog_dia = list(trayecto_est_prog_dia)
    for i in range(len(trayecto_est_prog_dia)):
        tabla_trayecto_est_prog_dia[i] = list(trayecto_est_prog_dia[i])


def Getdelaytime(estacion_ini,estacion_fin):  # Busca el tiempo de retraso en el diccionario de demandas y proporciones.
    '''
    Funcion para obtener el tiempo de demora en una estacion accediendo al diccionario demanda_dic, si no lo encuentra retorna 0.
    '''
    global demanda_dic
    try:
        return demanda_dic[int(estacion_ini), int(estacion_fin)]
    except:
        return 0

def convertHour(horas = None, hora = None):
    '''
    Funcion para imprimir un vector de horas en datetime.timedelta.
    '''
    if horas:
        horasD = []
        for i, x in enumerate(horas):
            horasD.append(str(x))
        return np.array(horasD)
    elif hora:
        horaStr = str(datetime.timedelta(seconds= int(round(((hora*24)*60)*60))))
        if len(horaStr) == 7:
            horaStr = "0" + horaStr
        return horaStr

def Conexion(db_host='127.0.0.1',usuario='root',clave='123456',base_de_datos='rutamega_principal'):
    '''
    Esta funcion se encarga de conectarse a una base de datos por medio de un cursor.
    los datos de conexion por defecto son root@127.0.0.1 con contraseña '123456' a la base de datos rutamega_principal.
    Hace consultas para obtener los datos necesarios de la base de datos a la que se acceda con el fin de ejecutar el control sobre el sistema.	
	'''
    global trayecto_est_prog_dia, velocidad_tiempo_distancia_linea, velocidad_tiempo_distancia_linea_dic, demanda_proporcion, demanda_dic

    db = MySQLdb.connect(host=db_host, user=usuario, passwd=clave, db=base_de_datos)
    cursor = db.cursor() 
    query_quotes = "SET sql_mode='ANSI_QUOTES'"  # Consulta para que las consultas acepte tablas con "caracteres especiales".
    cursor.execute(query_quotes)
    trayecto_est_prog_query = '''SELECT idtrayectoestacionprogramacion,idprogramaciontabladia,horallegadaprogramada,horallegadareal,idbus,modeloestadobus,modeloholding,modelovelocidad,tiempoproximaestacion,horallegadaestimada,idlinea,horasalidareal,horasalidaprogramada,idtrayectoestacion,idtrayectoestacionprogramaciondia FROM "trayecto-estacion_programacion_dia" ORDER BY `trayecto-estacion_programacion_dia`.`idtrayectoestacionprogramaciondia` ASC'''
    cursor.execute(trayecto_est_prog_query)
    trayecto_est_prog_dia = cursor.fetchall()
    velocidadtiempo = '''SELECT idlinea,velocidadteoricalinea,tiempolineateorico,distancialineateorica,idestacioninicio,idestacionfinal FROM "trayecto-linea"'''  ## para las velocidades y los tiempos en las lineas
    cursor.execute(velocidadtiempo)
    velocidad_tiempo_distancia_linea = cursor.fetchall()

    for i in range(len(velocidad_tiempo_distancia_linea)):
        '''
         Llenado del diccionario velocidad_tiempo_distancia_linea con la velocidad(1), el tiempo(2) y la distancia(3) de cada linea
         Ademas de esto tiene la estacion de inicio(4) y la estacion de finalizacion de cada linea(5).
        '''
        velocidad_tiempo_distancia_linea_dic[velocidad_tiempo_distancia_linea[i][0]] = [
            velocidad_tiempo_distancia_linea[i][1], velocidad_tiempo_distancia_linea[i][2],
            velocidad_tiempo_distancia_linea[i][3], velocidad_tiempo_distancia_linea[i][4],
            velocidad_tiempo_distancia_linea[i][5]]  # velocidad_tiempo_distancia_linea_dic[linea]=[velocidad,tiempo,distancia,estinicio,estfinal]

			
    tiempoestaciones = '''SELECT idestacionorigen,idestaciondestino,demanda FROM "estacion-matrices"'''  ### para tiempo en la estacion -> diferencia entre llegada y salida.
    cursor.execute(tiempoestaciones)
    tst = cursor.fetchall()
    demanda_proporcion = list(tst)
    
    for i in range(len(demanda_proporcion)):
        demanda_proporcion[i] = list(tst[i])
        '''
		Llena el diccionario demanda_dic con los valores de la demanda de una estacion a otra
        el diccionario tiene doble llave las cuales consistes en estacion de inicio y estacion de finalizacion.
        ademas de esto la demanda se multiplica por un factor que para este caso es 1/60 ( valor establecido por Diego).
       '''
        if demanda_proporcion[i][2]:
            demanda_dic[int(demanda_proporcion[i][0]), int(demanda_proporcion[i][1])] = demanda_proporcion[i][2] * 1 / 60  # Creacion diccionario con 0 : demanda. tiempo de abordaje y salida.
        else :
            demanda_dic[int(demanda_proporcion[i][0]), int(demanda_proporcion[i][1])] = 0
    return cursor, db


####

def Generarhoraestimada(): 
    '''
    Funcion para generar las horas estimadas faltantes de cada servicio y generar los valores de holding optimos por servicio.
    La generacion de valores optimos de holding por servicio se hace mediante la funcion optimizar.
    '''
    global tabla_trayecto_est_prog_dia, velocidad_tiempo_distancia_linea_dic
    #cursor, db = Conexion()
    #cursor,db=Conexion(db_host = 'admin.megaruta.co',usuario = 'rutamega_eqopt',clave ='eedd8ae977b7f997ce92aa1b0')
    cursor,db=Conexion(db_host = 'localhost',usuario = 'optimizacion',clave ='fdoq9zSyfSlMsyW9wGkh')
    base_de_datos = 'rutamega_principal'  #rutamega_principal
    Conversion_lista()
    tiempoentreestaciones                   = 0
    tiempodemora                            = 0
    posicion_tabla_tepd                     = 0
    tiempoLlegadaBusEstimado                = []
    tiempoAbordajeSalidaBuses               = []
    tiempoLlegadaBusProgramado              = []
    pesos                                   = []
    Servicio_actual                         = tabla_trayecto_est_prog_dia[0][1]
    bandera                                 = 1
    error                                   = 0


    while (Servicio_actual == tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1] ):
        horaestimada                                = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][9]
        hora_llegada_real                           = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][3]
        tiempollegadaestimado			    		= tabla_trayecto_est_prog_dia[posicion_tabla_tepd][8]
        if not hora_llegada_real:
            '''
            Si no hay hora real despues de estar chequeando, indica que está en la linea donde el articulado se encuentra.
            '''
            if horaestimada :
                if bandera ==1 : # Si la bandera esta subida, se baja y se agrega la horaestimada a el vector que se enviara a Pyomo.
                    bandera=0
                    tiempoLlegadaBusEstimado.append(horaestimada)
                peso                                    = '''SELECT pesocontrol FROM "estacion" WHERE idestacion=(SELECT idestacion FROM "trayecto-estacion" WHERE idtrayectoestacion=%i)''' % \
                                                                tabla_trayecto_est_prog_dia[posicion_tabla_tepd][13]  # Se lee el valor del peso para la estacion de la posicion actual en la tabla.
                cursor.execute(peso)
                auxiliar                                = cursor.fetchone()
                if tabla_trayecto_est_prog_dia[posicion_tabla_tepd][10]:# Si tiene valor en idlinea en la tabla trayecto-estacion_programacion_dia
                    error=0
                    linea                                   = int(tabla_trayecto_est_prog_dia[posicion_tabla_tepd][10]) 
                    estacion_ini, estacion_fin              = velocidad_tiempo_distancia_linea_dic[linea][3], velocidad_tiempo_distancia_linea_dic[linea][4]
                    tiempodemora                            = Getdelaytime(estacion_ini,estacion_fin)         # sacar demanda*1/60 ( tiempo de retraso) #Tiempo Abordaje Salida Buses
                    '''
                     Toma de datos para el modelo de optimizacion en pyomo.
                    '''
                    horallegadaprogramada                   = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][2]                              # tiempoSalidaEstacionProgramada
                    tiempoLlegadaBusProgramado.append(horallegadaprogramada)
                    tiempoAbordajeSalidaBuses.append(tiempodemora)                                                                              # Agregar el tiempo de demora entre estaciones al vector que se enviara a pyomo.
                    valor_peso                              = auxiliar[0]                                                                       # Valor del peso de control para la estacion actual en el recorrido.
                    if valor_peso : # Si encontro un valor con la consulta al peso de la estacion siguiente lo agrega, de lo contrario agrega 0.
					  pesos.append(valor_peso)
                    else :
                      pesos.append(0)
                else: # Si no tiene valor en idlinea, levanta la bandera error.
                    error=1
		    
                posicion_tabla_tepd                     = posicion_tabla_tepd + 1 # Actualizacion del
                if posicion_tabla_tepd < len(tabla_trayecto_est_prog_dia):

                    if tabla_trayecto_est_prog_dia[posicion_tabla_tepd][10]:# Si tiene valor en idlinea.
                      error=0
                      linea                                   = int(tabla_trayecto_est_prog_dia[posicion_tabla_tepd][10]) #
                      estacion_ini, estacion_fin              = velocidad_tiempo_distancia_linea_dic[linea][3], velocidad_tiempo_distancia_linea_dic[linea][4]
                      tiempoentreestaciones                   = velocidad_tiempo_distancia_linea_dic[linea][1]  # tiempo que hay entre estaciones sin retraso
                      tiempollegadaestimado		    		= tiempollegadaestimado + tiempoentreestaciones + tiempodemora
                      horaestimada                            = horaestimada + datetime.timedelta(seconds=tiempoentreestaciones + tiempodemora)   # tiempoLlegadaBusEstimado
                      tiempoLlegadaBusEstimado.append(horaestimada)  # Agregar la hora estimada al vector que se enviara a pyomo.

                    '''
                    Fin toma de datos.
                 '''

                    if Servicio_actual == tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1] and not error: #Si la posicion actual es menor a la cantidad de posiciones y no hubo error.
                        tabla_trayecto_est_prog_dia[posicion_tabla_tepd][9] = horaestimada 
                        tabla_trayecto_est_prog_dia[posicion_tabla_tepd][8] = tiempollegadaestimado 

                    elif not error:  # Si no la bandera error esta en 0 se cambia el valor del servicio actual.
                        '''
                        Se genera la optimizacion del servicio,para asi hallar el valor de holding para cada estacion del servicio.
                    '''
                        tiempoLlegadaBusEstimado.pop() # Problema : Cuando es mas de un servicio agrega un valor de mas al vector, puede generar problemas en la funcion optimizacion.
                        optimizacion(tiempoLlegadaBusEstimado, Servicio_actual,tiempoLlegadaBusProgramado, pesos)
                        #printopt(tiempoLlegadaBusEstimado, Servicio_actual,tiempoLlegadaBusProgramado, pesos)
                        Servicio_actual                 = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1]
                        # Re inicializacion de los vectores que se cargaran con los valores necesarios para generar el modelo en Pyomo.
                        tiempoLlegadaBusEstimado        = []
                        tiempoAbordajeSalidaBuses       = []
                        tiempoLlegadaBusProgramado  	= []
                        pesos                           = []
                        horaestimada                    = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][9]
                        hora_llegada_real                           = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][3]
                        if not hora_llegada_real: # Si no hay hora real en el siguiente valor, se pone la bandera en 0 y se agrega ese valor a el vector de horaestimada del siguiente servicio.
                            bandera						=0
                            tiempoLlegadaBusEstimado.append(horaestimada)
                        else: # Si hay hora real se sube la bandera.
                            bandera						=1
                        
                else:
                    #Se genera la optimizacion del ultimo servicio, con los nuevos valores se actualizan los holdings.
                    optimizacion(tiempoLlegadaBusEstimado, Servicio_actual, tiempoLlegadaBusProgramado,pesos)  
                    #printopt(tiempoLlegadaBusEstimado, Servicio_actual, tiempoLlegadaBusProgramado,pesos)  
                    break
            else : # Si no tiene hora de llegada programada , unicamente aumenta la posicion en la tabla y verifica si debe cambiar el valor del servicio actual.
                posicion_tabla_tepd    = posicion_tabla_tepd+1
                if posicion_tabla_tepd < len (trayecto_est_prog_dia):
                    if Servicio_actual != tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1]:
                        Servicio_actual = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1]
                else:
                    break
        else :  # Si tiene hora de llegada real , unicamente aumenta la posicion en la tabla y verifica si debe cambiar el valor del servicio actual.
                posicion_tabla_tepd    = posicion_tabla_tepd+1
                if posicion_tabla_tepd < len (trayecto_est_prog_dia):
                    if Servicio_actual != tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1]:
                        Servicio_actual = tabla_trayecto_est_prog_dia[posicion_tabla_tepd][1]
                else :
                    break
    return cursor, db

def printopt(tiempoLlegadaBusEstimado, Servicio, tiempoLlegadaBusProgramado, pesos):
    '''
    Funcion que trabaja con todos los holding en 0 unicamente para ver el valor de la funcion objetivo y funcionamiento en general.
    '''
    global servicio_holding
    #print len(tiempoLlegadaBusEstimado),len(tiempoLlegadaBusProgramado),len(pesos)
    #print "Llegadaprogramada",convertHour(horas = tiempoLlegadaBusProgramado)
    #print "Estimado",convertHour(horas = tiempoLlegadaBusEstimado)
    holding=[]
    if len(tiempoLlegadaBusEstimado)==len(tiempoLlegadaBusProgramado)==len(pesos):
      for h in range(len(tiempoLlegadaBusProgramado)):
          holding.append(0)
      print sum(((tiempoLlegadaBusEstimado[e].total_seconds() + (
              holding[e] * pesos[e]) - tiempoLlegadaBusProgramado[
                           e].total_seconds())) ** 2 for e in range(len(tiempoLlegadaBusProgramado)))
      servicio_holding[Servicio]=holding[0]

def Get_holding(Servicio):
    '''
    
    '''
    global servicio_holding
    try:
	   return servicio_holding[Servicio]
    except :
       return 0

def optimizacion(tiempoLlegadaBusEstimado,Servicio,tiempoLlegadaBusProgramado, pesos):
    '''
        Funcion que se encarga de crear el modelo en Pyomo y mandarlo al solver IPOPT(Interior Point OPTimizer) para minimizar.
        IMPORTANTE : Optimizacion : Funcion objetivo es --> hora_salidaprogramada+(holding*proporcion)-horaestimadallegada  (todo dado en segundos).
    '''
    global servicio_holding
    #print len(tiempoLlegadaBusEstimado),len(tiempoLlegadaBusProgramado),len(pesos)
    #print "Estimado",convertHour(horas = tiempoLlegadaBusEstimado)
    #print "Salidaprogramada",convertHour(horas = tiempoSalidaEstacionProgramada)
    if len(tiempoLlegadaBusEstimado)==len(tiempoLlegadaBusProgramado)==len(pesos):
	    f = open('/etc/optimizacion/archivos/GenerarControl.dat', 'w')
	    f.write("param estaciones := %s ;" % len(tiempoLlegadaBusProgramado))
	    f.close()
	    model                       = AbstractModel()
	    model.estaciones            = Param(within=NonNegativeIntegers)
	    model.numE                  = RangeSet(0, model.estaciones - 1)  # Estaciones en el servicio.
	    model.holding               = Var(model.numE, initialize=0)

	    def obj_rule(model): # Funcion objetivo : hora llegada estimada + (holding*peso) - hora llegada programada  (todo dado en segundos)
		return sum(((tiempoLlegadaBusEstimado[e].total_seconds() + (
		    model.holding[e]*pesos[e])- tiempoLlegadaBusProgramado[
		                 e].total_seconds()))**2  for e in model.numE)

#bounds = (-30,30)
	    model.obj                   = Objective(rule=obj_rule, sense=minimize)

	    def holding_constraint_rule(model, r):
		return ( -30,model.holding[r], 30)
	    
	    model.holdingConstraint     = Constraint(model.numE, rule=holding_constraint_rule)
	    
	    from coopr.opt import SolverFactory
	    instance                    = model.create("/etc/optimizacion/archivos/GenerarControl.dat");  # Se crea una instancia del modelo, con los datos del archivo.dat ya que es un modelo abstracto
	    opt                         = SolverFactory("ipopt")  # Se utiliza el solver ipopt ya que es un problema no lineal
	    results                     = opt.solve(instance)  # Resuelve el modelo y devuelve su solución
	    results.write()

	    instance.load(results)
            servicio_holding[Servicio]=instance.holding[0].value



def GenerarControl(cursor=None,valor_velocidad_arbitrario=0):  # Funcion para generar el control sobre el sistema en funcionamiento.
    global tabla_trayecto_est_prog_dia,servicio_holding
    contador_holding                = 0
    valor_velocidad_arbitrario      = valor_velocidad_arbitrario
    Servicio_actual                 = -1
    "Inicializacion de variables que se van a guardar. Evita problemas."
    modeloholding                   = 0
    modeloestadoact                 = "Sin tiempo llegada estimado."
    modelovelocidad                 = 0
    actual_servicio                 = 1 # Bandera para indicar si el registro actual en la tabla trayecto-estacion_programacion_dia es el actual de un servicio.
    
    print servicio_holding
    for i in range(len(tabla_trayecto_est_prog_dia)):
	if tabla_trayecto_est_prog_dia[i][10]:
		linea = int(tabla_trayecto_est_prog_dia[i][10])
		## Se mira que exista velocidad de linea, si no existe o el valor es invalido se mira la velocidad de la ruta.
		if velocidad_tiempo_distancia_linea_dic[linea][0] > 0 and velocidad_tiempo_distancia_linea_dic[linea][0]:
		    modelovelocidad             = velocidad_tiempo_distancia_linea_dic[linea][0]
		else:
		    velocidadruta               = '''SELECT velocidadteoricaruta FROM ruta WHERE idruta=(SELECT idruta FROM trayecto WHERE idtrayecto=(SELECT idtrayecto FROM "trayecto-linea" WHERE idlinea=%s)) ''' % linea
		    cursor.execute(velocidadruta)  # Buscar la velocidad de la ruta
		    Vr                          = cursor.fetchone()
		    modelovelocidad             = Vr[0]  # Se lee el valor encontrado con la consulta SQL.
		
        tiempoestimado		        = tabla_trayecto_est_prog_dia[i][8]
        hora_estimada               = tabla_trayecto_est_prog_dia[i][9]
        hora_programada             = tabla_trayecto_est_prog_dia[i][2]
        hora_llegada_real           = tabla_trayecto_est_prog_dia[i][3]
        hora_salida_real            = tabla_trayecto_est_prog_dia[i][11]


        if hora_llegada_real and hora_salida_real:  # Si ya llego y ya salio de la estacion actual.
            modeloestadoact = "Salio a las %s." % str(hora_salida_real).split(" ")[1]
        elif hora_llegada_real and not hora_salida_real:  # Si llego pero no ha salido de la estacion actual.
            modeloestadoact = "En estacion."
        elif not hora_llegada_real and not hora_salida_real :
####################### Aca esta metiendose por el "else" que implica no cambiar el valor de modeloestadoact.
            if tiempoestimado != None :
               modeloestadoact = "En %d segundos." % tiempoestimado

        ###
        updateestadoact     = '''UPDATE `trayecto-estacion_programacion_dia` SET modeloestadobus='%s' WHERE idtrayectoestacionprogramaciondia = %i ''' % \
                                             (str(modeloestadoact), tabla_trayecto_est_prog_dia[i][14])
        cursor.execute(updateestadoact)
        ####
        if not hora_llegada_real:  # Si el bus no tiene hora de llegada real ( no ha pasado por esa linea).

            if hora_programada:  # Evita inconsistencias.

                if hora_estimada:  # Si el bus tiene hora estimada (evita recorrer los servicios que aun no estan en funcionamiento).
                    if Servicio_actual != tabla_trayecto_est_prog_dia[i][1]:
                       actual_servicio   = 1
                       Servicio_actual   = tabla_trayecto_est_prog_dia[i][1]
                    else:
                       actual_servicio   = 0
                    modeloholding=Get_holding(Servicio_actual)
                    auxiliar_horallegadaestimada=hora_estimada.total_seconds()+modeloholding
                    nueva_horallegadaestimada=datetime.timedelta(seconds=auxiliar_horallegadaestimada)
                    if not actual_servicio :
                        updatehorallegadaestimada= '''UPDATE `trayecto-estacion_programacion_dia` SET horallegadaestimada='%s',tiempoproximaestacion=%s WHERE idtrayectoestacionprogramaciondia = %i ''' % \
                                             (nueva_horallegadaestimada,tabla_trayecto_est_prog_dia[i][8]+modeloholding, tabla_trayecto_est_prog_dia[i][14])
		        cursor.execute(updatehorallegadaestimada)

                    if modeloholding > 0.1 :
                           updatevelocidad   = '''UPDATE `trayecto-estacion_programacion_dia` SET modeloholding=%s,modelovelocidad=%s WHERE idtrayectoestacionprogramaciondia = %i ''' % \
                                             (modeloholding,modelovelocidad-valor_velocidad_arbitrario, tabla_trayecto_est_prog_dia[i][14])

                    elif modeloholding < 0.1 :
                            updatevelocidad   = '''UPDATE `trayecto-estacion_programacion_dia` SET modeloholding=%s,modelovelocidad=%s WHERE idtrayectoestacionprogramaciondia = %i ''' % \
                                             (modeloholding,modelovelocidad+valor_velocidad_arbitrario, tabla_trayecto_est_prog_dia[i][14])

                    else:
                            updatevelocidad   = '''UPDATE `trayecto-estacion_programacion_dia` SET modeloholding=%s,modelovelocidad=%s WHERE idtrayectoestacionprogramaciondia = %i ''' % \
                                             (modeloholding,modelovelocidad, tabla_trayecto_est_prog_dia[i][14])

                    cursor.execute(updatevelocidad)  #actualizacion sobre el cursor de la base de datos.
    db.commit() # Se hace commit a la base de datos para reflejar los cambios hechos en el cursor.

cursor, db = Generarhoraestimada()
GenerarControl(cursor,1)
cursor.close()
db.close()

