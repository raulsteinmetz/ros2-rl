def generate_target_sdf(x, y, z):
        return f"""
        <?xml version='1.0'?>
        <sdf version='1.6'>
        <model name='target_mark'>
            <static>true</static>  <!-- This makes the model static -->
            <pose>{x} {y} {z} 0 0 0</pose>
            <link name='link'>
            <visual name='visual'>
                <geometry>
                <plane><normal>0 0 1</normal><size>0.5 0.5</size></plane>  <!-- Smaller size -->
                </geometry>
                <material>
                <ambient>1 0 0 1</ambient>  <!-- Bright red color for visibility -->
                </material>
            </visual>
            </link>
        </model>
        </sdf>
        """