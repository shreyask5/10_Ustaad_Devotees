class EmergencyContact {
  final String name;
  final String number;

  EmergencyContact({
    required this.name,
    required this.number,
  });
}

class GuidanceItem {
  final String title;
  final String subtitle;
  final String content;
  final String? videoUrl;
  final List<EmergencyContact> emergencyContacts;

  GuidanceItem({
    required this.title,
    required this.subtitle,
    required this.content,
    this.videoUrl,
    this.emergencyContacts = const [],
  });
}

class GuidanceService {
  Future<List<GuidanceItem>> getGuidanceContent() async {
    // This is a placeholder implementation
    // In a real app, this would fetch data from an API or local storage
    return [
      GuidanceItem(
        title: 'Self-Defense Basics',
        subtitle: 'Essential self-defense techniques for emergency situations',
        content: '''
1. Stay aware of your surroundings
2. Trust your instincts
3. Maintain a confident posture
4. Keep your phone easily accessible
5. Know your escape routes
''',
        videoUrl: 'https://example.com/self-defense-video',
        emergencyContacts: [
          EmergencyContact(
            name: 'Police Emergency',
            number: '100',
          ),
          EmergencyContact(
            name: 'Women\'s Helpline',
            number: '1091',
          ),
        ],
      ),
      GuidanceItem(
        title: 'Legal Rights',
        subtitle: 'Know your legal rights and protections',
        content: '''
1. Right to file a complaint
2. Protection against harassment
3. Right to self-defense
4. Legal recourse for stalking
5. Workplace safety rights
''',
        emergencyContacts: [
          EmergencyContact(
            name: 'Legal Aid',
            number: '1800-XXX-XXXX',
          ),
        ],
      ),
      GuidanceItem(
        title: 'Emergency Preparedness',
        subtitle: 'How to prepare for emergency situations',
        content: '''
1. Keep emergency contacts updated
2. Share your location with trusted contacts
3. Have a safety plan
4. Know nearby safe zones
5. Keep important documents accessible
''',
        videoUrl: 'https://example.com/emergency-preparedness',
      ),
    ];
  }
} 