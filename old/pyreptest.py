from pyrep import PyRep

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('/home/raul/install/coppelia/scenes/mobile_and_wall.ttt') 
pr.start()  # Start the simulation

# Do some stuff

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application