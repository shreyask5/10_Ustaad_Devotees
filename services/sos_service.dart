import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'location_service.dart';

class SOSService {
  final FirebaseMessaging _messaging = FirebaseMessaging.instance;

  Future<void> sendEmergencyAlert({
    required LocationData location,
    required String message,
  }) async {
    // Send push notification to emergency contacts
    await _sendPushNotification(message, location);

    // Send SMS to emergency contacts (implement your SMS gateway)
    await _sendSMS(message, location);

    // Call emergency services API (implement your emergency services API)
    await _callEmergencyServices(location);
  }

  Future<void> _sendPushNotification(String message, LocationData location) async {
    // Implement Firebase Cloud Messaging logic here
    // This is a placeholder for the actual implementation
    await _messaging.send(
      Message(
        data: {
          'type': 'emergency',
          'message': message,
          'latitude': location.latitude.toString(),
          'longitude': location.longitude.toString(),
        },
        notification: Notification(
          title: 'Emergency Alert',
          body: message,
        ),
      ),
    );
  }

  Future<void> _sendSMS(String message, LocationData location) async {
    // Implement SMS sending logic here
    // This is a placeholder for the actual implementation
    final url = Uri.parse('YOUR_SMS_GATEWAY_API');
    await http.post(
      url,
      body: json.encode({
        'message': message,
        'location': {
          'latitude': location.latitude,
          'longitude': location.longitude,
        },
      }),
    );
  }

  Future<void> _callEmergencyServices(LocationData location) async {
    // Implement emergency services API call here
    // This is a placeholder for the actual implementation
    final url = Uri.parse('YOUR_EMERGENCY_SERVICES_API');
    await http.post(
      url,
      body: json.encode({
        'type': 'emergency',
        'location': {
          'latitude': location.latitude,
          'longitude': location.longitude,
        },
      }),
    );
  }
} 