import os
def create_mitsuba_scene(obj_path, output_xml):
    scene_content = f"""<scene version="3.0.0">
    <!-- Configuration du moteur de rendu -->
    <integrator type="path"/>

    <!-- Définition du maillage (Mesh) -->
    <shape type="obj">
        <string name="filename" value="{obj_path}"/>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
        </bsdf>
    </shape>

    <!-- Source de lumière (Lumière distante/Soleil) -->
    <emitter type="directional">
        <vector name="direction" value="1, -1, 1"/>
        <rgb name="irradiance" value="1.0, 1.0, 1.0"/>
    </emitter>

    <!-- Configuration de la Caméra -->
    <sensor type="perspective">
        <float name="fov" value="45"/>
        <transform name="to_world">
            <lookat target="0, 0, 0" origin="0, 5, 10" up="0, 1, 0"/>
        </transform>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
        </film>
        <sampler type="independent">
            <integer name="sample_count" value="64"/>
        </sampler>
    </sensor>
</scene>
"""
    with open(output_xml, "w", encoding="utf-8") as f:
        os.makedirs(os.path.dirname(output_xml), exist_ok=True)
        f.write(scene_content)
    print(f"✅ Scène Mitsuba créée : {output_xml}")

# Utilisation
obj_path = "scenes/corridor/corridor.obj"  # path to your OBJ file
output_xml = "scenes/corridor/corridor2.xml"  # desired output XML file

create_mitsuba_scene(obj_path, output_xml)