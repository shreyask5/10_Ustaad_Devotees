import 'package:flutter/material.dart';
import '../services/sos_service.dart';
import '../services/location_service.dart';
import '../widgets/alert_dialogs.dart';

class SOSScreen extends StatefulWidget {
  const SOSScreen({super.key});

  @override
  State<SOSScreen> createState() => _SOSScreenState();
}

class _SOSScreenState extends State<SOSScreen> {
  final SOSService _sosService = SOSService();
  final LocationService _locationService = LocationService();
  bool _isLoading = false;

  Future<void> _triggerSOS() async {
    final bool? confirmed = await showSOSConfirmationDialog(context);
    if (confirmed == true) {
      setState(() => _isLoading = true);
      try {
        final location = await _locationService.getCurrentLocation();
        await _sosService.sendEmergencyAlert(
          location: location,
          message: 'Emergency! I need help!',
        );
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Emergency alert sent successfully!'),
              backgroundColor: Colors.green,
            ),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error sending alert: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      } finally {
        if (mounted) {
          setState(() => _isLoading = false);
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('SOS Emergency'),
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'Press the button below to send an emergency alert',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 18.0),
            ),
            const SizedBox(height: 32.0),
            _isLoading
                ? const CircularProgressIndicator()
                : ElevatedButton(
                    onPressed: _triggerSOS,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 48.0,
                        vertical: 24.0,
                      ),
                    ),
                    child: const Text(
                      'SOS',
                      style: TextStyle(
                        fontSize: 24.0,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
          ],
        ),
      ),
    );
  }
} 