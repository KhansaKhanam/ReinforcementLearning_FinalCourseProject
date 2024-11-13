import traci
import sumolib
import sys

nets_file = 

try:
    # Define paths to SUMO binary, network file, and route file
    sumo_binary = "sumo-gui"  # Use "sumo" for non-GUI mode
    # nets_file = "path/net.xml"  # Replace with actual path to your net.xml
    # routes_file = "path/to/rou.xml"  # Replace with actual path to your rou.xml

    # Command to start SUMO with TraCI connection
    sumo_cmd = [sumo_binary, "-n", nets_file, "-r", routes_file, "--start"]
    
    # Attempt to start SUMO with the specified command
    try:
        traci.start(sumo_cmd, label="sim1")
    except traci.exceptions.TraCIException as e:
        print(f"Failed to start TraCI: {e}")
        sys.exit(1)

    print("TraCI connected successfully.")
    
    # Main simulation loop with safe TraCI operations
    for step in range(1000):
        try:
            traci.simulationStep()  # Advance simulation by one step
            
            # Check vehicle count on a specific edge
            edge_id = "t_e"  # Replace with actual edge ID
            try:
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                print(f"Step {step}, Vehicle count on {edge_id}: {vehicle_count}")
            except traci.exceptions.TraCIException as e:
                print(f"Error retrieving vehicle count for edge {edge_id}: {e}")
            
            # Get traffic light phase and queue length
            tls_id = "t"  # Replace with actual traffic light ID
            try:
                phase = traci.trafficlight.getPhase(tls_id)
                print(f"Traffic light {tls_id} phase: {phase}")
            except traci.exceptions.TraCIException as e:
                print(f"Error retrieving traffic light phase for {tls_id}: {e}")
            
            # Pause to inspect values for each step
            input("Press Enter to continue to the next step...")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            break  # Safely exit the loop if user interrupts

except (FileNotFoundError, traci.exceptions.TraCIException) as e:
    print(f"Error in setup or simulation: {e}")

except Exception as e:
    print(f"Unexpected error occurred: {e}")

finally:
    # Ensure that TraCI closes safely even if an error occurs
    try:
        if traci.isLoaded():
            traci.close()
            print("TraCI connection closed safely.")
    except traci.exceptions.TraCIException as e:
        print(f"Error while closing TraCI: {e}")
