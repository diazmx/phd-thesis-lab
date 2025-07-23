CREATE TABLE tecnico (
    id_tecnico INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    curp varchar(18) NOT NULL UNIQUE,
    nombre varchar(50) NOT NULL,
    apellido varchar(50),
    fecha_nacimiento DATE
);

CREATE TABLE telefono (
    id_telefono INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    id_tecnico_fk INT,
    telefono1 varchar(20),
    telefono2 varchar(20),
    FOREIGN KEY (id_tecnico_fk) REFERENCES tecnico(id_tecnico)
);

CREATE TABLE planta (
    id_planta INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    color varchar(50) NOT NULL UNIQUE,
    superficie DECIMAL(10, 2)
);

CREATE TABLE maquina (
    id_maquina INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    id_maquina_respuesto INT,
    id_planta_fk INT,
    esta_servicio BOOLEAN DEFAULT TRUE, 
    marca varchar(50),
    modelo varchar(50),
    FOREIGN KEY (id_maquina_respuesto) REFERENCES maquina(id_maquina),
    FOREIGN KEY (id_planta_fk) REFERENCES planta(id_planta)
);

CREATE TABLE opera (
    id_opera INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    id_tecnico_fk INT,
    id_maquina_fk INT,
    turno ENUM('Dia', 'Noche', 'Tarde'),
    fecha_inicio DATE,
    fecha_final DATE,
    FOREIGN KEY (id_tecnico_fk) REFERENCES tecnico(id_tecnico),
    FOREIGN KEY (id_maquina_fk) REFERENCES maquina(id_maquina)
)

CREATE TABLE proceso (
    id_proceso INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    id_planta_fk INT,
    nombre varchar(100),
    complejidad ENUM('Facil', 'Medio', 'Dificil'),
    FOREIGN KEY (id_planta_fk) REFERENCES planta(id_planta)
)